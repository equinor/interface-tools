import os
import tempfile

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.fileshare import ShareFileClient


class DataHandlerAzureFileShare:
    def __init__(
        self,
        share_name: str,
        storage_account: str = None,
        connection_string: str = None,
    ) -> None:
        self.share_name = share_name
        self.storage_account = storage_account
        self.connection_string = connection_string

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Read an excel file from local or from fileshare
        :param file_path: path of the file
        :param storage_account: name of the storage account
        :param share_name: name of the file share
        """
        if self.connection_string:
            file_client = ShareFileClient.from_connection_string(
                conn_str=self.connection_string,
                share_name=self.share_name,
                file_path=file_path,
            )
        else:
            file_client = ShareFileClient(
                account_url=f"https://{self.storage_account}.file.core.windows.net/",
                credential=DefaultAzureCredential(),
                share_name=self.share_name,
                file_path=file_path,
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv.gz", delete=False
        ) as temp:
            file_client.download_file().readinto(temp)
            df = pd.read_csv(temp.name, compression="gzip")
        os.remove(temp.name)
        return df
