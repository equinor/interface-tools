import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.model import InferenceConfig, Model
from azureml.core.run import Run
from azureml.core.webservice.aci import AciWebservice
from azureml.core.webservice.local import LocalWebservice

logger = logging.getLogger()


class DeploymentTarget(Enum):
    LOCAL = "LOCAL"
    ACI = "ACI"
    AKS = "AKS"


class AzureML:
    """
    This is the Azure ML class that handles most interfacing between the Azure ML platform
    and you!

    Some things to note:

    Requirements when running locally
    --------------------------------------
        - (REQUIRED) Docker client needs to be installed on your local machine
        - (REQUIRED) On Windows, the folder "C:\\Users\\USERNAME\\AppData\\Local\\Temp\\azureml_runs" needs to
                     be included as a resource under *File sharing* in the Docker client
    """

    def __init__(
        self,
        name: str,
        script: str,
        source_directory: str,
        environment_name: str,
        docker_base_image: str,
        run_local: bool = True,
        cluster_name: Optional[str] = None,
        cluster_config: Dict[str, Union[str, int]] = None,
        environment_variables: Dict[str, str] = None,
        enable_docker: bool = True,
    ):
        """Class constructor for Azure ML interface module

        :param name: Name of experiment and model
        :type name: str
        :param script: Path to the ``train.py`` file relative to the source directory
        :type script: str
        :param source_directory: Path to the project source directory. This will normally the top-level directory
        :type source_directory: str
        :param environment_name: Name of the Azure ML environment
        :type environment_name: str
        :param docker_base_image: Name of the base Docker image
        :type docker_base_image: str
        :param run_local: Defines whether to run the experiment locally or not, defaults to True
        :type run_local: bool, optional
        :param cluster_name: Name of Azure ML cluster to train on, this overrules ``run_local``. Defaults to None
        :type cluster_name: Optional[str], optional
        :param cluster_config: Config for creating a new cluster to train on, this overrules ``run_local`` and ``cluster_name``. Defaults to None
        :type cluster_config: Dict[str, Union[str, int]], optional
        :param environment_variables: Defines environment variables available in an experiment run
        :type cluster_config: Dict[str, str], optional
        """
        self.name = name
        self.script = script
        self.source_directory = source_directory
        self.environment_name = environment_name
        self.docker_base_image = docker_base_image
        self.run_local = run_local
        self.cluster_name = cluster_name
        self.cluster_config = cluster_config
        self.environment_variables = environment_variables
        self.enable_docker = enable_docker
        self.run: Optional[Run] = None
        self.model = None
        self.ws = None

    def get_workspace(self) -> Workspace:
        """Get the workspace from Azure ML

        :return: Azure ML workspace handle
        :rtype: Workspace
        """
        if self.ws:
            return self.ws
        self.ws = Workspace.from_config()
        return self.ws

    def get_environment(self) -> Environment:
        """Gets the environment in Azure ML and merges local config file

        :return: Azure ML environment handle
        :rtype: Environment
        """
        self.get_workspace()
        if self.environment_name in Environment.list(self.ws).keys():
            self.environment = Environment.get(self.ws, self.environment_name)
        else:
            self.environment = Environment(self.environment_name)
        self.environment = self._set_environment_properties(self.environment)
        return self.environment

    def register_environment(self) -> Any:
        """Builds and registers a new environment in Azure ML according to local config file

        :return: Environment build information
        :rtype: Any
        """
        self.get_workspace()
        self.environment = Environment(self.environment_name)
        self.environment = self._set_environment_properties(self.environment)
        build = self.environment.build(self.ws)
        self.environment.register(self.ws)
        return build

    def deploy(
        self,
        name: str,
        inference_config: Dict,
        deployment_config: Optional[Dict] = None,
        deployment_target: Union[DeploymentTarget, str] = DeploymentTarget.LOCAL,
        port: int = 8090,
        show_output: bool = True,
    ) -> LocalWebservice:
        """Host the model as a local web service or in AKS/ACI

        :param name: Name of the model to be loaded
        :type name: str
        :param inference_config: Inference config
        :type inference_config: Dict
        :param deployment_config: Deployment config
        :type deployment_config: Dict
        :param deployment_target: Deployment target
        :type deployment_target: DeploymentTarget
        :param port: Port endpoint number, defaults to 8090
        :type port: int, optional
        :param show_output: Show output log
        :type show_output: Boolean, default: True
        :return: Web service handle
        :rtype: LocalWebservice
        """
        if type(deployment_target) == str:
            deployment_target = DeploymentTarget(deployment_target)

        if self.model is None:
            raise ValueError("A model has to be registered before deploying")

        deployment = deployment_config
        if deployment_target == DeploymentTarget.LOCAL:
            deployment = LocalWebservice.deploy_configuration(port=port)
        elif deployment_target == DeploymentTarget.ACI:
            if deployment_config is None:
                raise ValueError(
                    "Deployment config needs to be specified if deploying to ACI"
                )
            deployment = AciWebservice.deploy_configuration()
        else:
            raise NotImplementedError("AKS target is not currently implemented")

        inference = InferenceConfig(**inference_config)
        inference.environment = self.environment

        service = Model.deploy(
            self.ws,
            name,
            [self.model],
            inference,
            deployment,
            overwrite=True,
        )
        service.wait_for_deployment(show_output=show_output)
        return service

    def register_model(self, tags: Optional[Dict[str, str]] = None) -> Model:
        """Register an Azure ML model

        :param tags: Tags that will be attached to an experiment run and model version, defaults to None
        :type tags: Optional[Dict[str, str]], optional
        :return: Azure ML model handle
        :rtype: Model
        """
        if self.run is None:
            raise ValueError(
                "An experiment has to be run before a model can be registered"
            )
        self.model = self.run.register_model(
            model_name=self.name,
            model_path="outputs/model.pkl",
            properties=tags,
        )
        return self.model

    def build_experiment(self) -> Any:
        """Build an experiment

        :return: Azure ML experiment run handle
        :rtype: Run
        """
        self.get_workspace()

        self.experiment = Experiment(workspace=self.ws, name=self.name)
        return self.experiment

    def run_experiment(
        self, tags: Optional[Dict[str, str]] = None, show_output: bool = True
    ) -> Any:
        """Run an Azure ML experiment

        :param tags: Tags that will be attached to an experiment run and model version, defaults to None
        :type tags: Optional[Dict[str, str]], optional
        :param show_output: Show output log
        :type show_output: Boolean, default: True
        :return: Azure ML experiment run handle
        :rtype: Run
        """
        self.get_workspace()
        self.get_environment()
        self.build_experiment()

        compute_target: Union[str, ComputeTarget] = "local"
        if not self.run_local:
            try:
                compute_target = ComputeTarget(
                    workspace=self.ws, name=self.cluster_name
                )
                logger.info("Found existing compute target.")
            except ComputeTargetException:
                logger.info("Creating a new compute target...")
                compute_config = AmlCompute.provisioning_configuration(
                    **self.cluster_config
                )
                compute_target = ComputeTarget.create(
                    self.ws, "sharedcluster", compute_config
                )
                compute_target.wait_for_completion(show_output=True)
            logger.debug(compute_target.get_status().serialize())

        src = ScriptRunConfig(
            source_directory=self.source_directory, script=self.script
        )
        src.run_config.environment = self.environment
        src.run_config.target = compute_target
        self.run = self.experiment.submit(src, tags=tags)
        self.run.wait_for_completion(show_output=show_output)
        self.run.get_environment()

        logger.info("Experiment completed successfully.")
        return self.run

    def _set_environment_properties(self, environment: Environment) -> Environment:
        environment.python.user_managed_dependencies = True
        environment.docker.enabled = self.enable_docker
        environment.docker.base_image = self.docker_base_image
        environment.environment_variables = self.environment_variables
        environment.inferencing_stack_version = "latest"
        return environment
