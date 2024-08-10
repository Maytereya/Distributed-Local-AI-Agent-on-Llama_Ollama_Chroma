import paramiko


class ParamikoFiles:
    """Класс для работы с файлом сохранения названия коллекции"""

    def __init__(self):
        # Укажите данные для подключения к серверу
        self.hostname = "46.0.234.32"
        self.port = 22
        self.username = "vl"
        self.password = "iDQFA1603"

        # Укажите путь и имя файла на сервере
        self.remote_filepath = "/home/vl/r_files/collection_name.txt"
        self.local_filepath = "collection_name.txt"

        # Содержимое файла, который нужно создать на сервере
        self.file_content = "Тестовый файл R-file"

    # def create_file(self, local_filepath):
    #     self.remote_filepath = "/home/vl/r_files/" + local_filepath
    #     self.local_filepath = local_filepath

    #     тут нужно дописать код вставления содержимого файла для эмбеддинга

    def write_to_file(self, content):
        """То, что мы запишем в файл"""
        self.file_content = content

    def create_and_save_file(self):
        """Создание и сохранение файла на сервере"""
        try:
            # Установление SSH соединения
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, self.port, self.username, self.password)

            # Использование SFTP для работы с файлами
            sftp = ssh.open_sftp()
            with sftp.file(self.remote_filepath, 'w') as remote_file:
                # тут запись в файл
                remote_file.write(self.file_content)

            sftp.close()
            ssh.close()
            print("File created and saved on the server successfully.")

        except Exception as e:
            print(f"Failed to create and save file: {e}")

    def download_file(self):
        """Загрузка файла с сервера и чтение его содержимого - имени коллекции"""
        try:
            # Установление SSH соединения
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, self.port, self.username, self.password)

            # Использование SFTP для работы с файлами
            sftp = ssh.open_sftp()
            sftp.get(self.remote_filepath, self.local_filepath)

            sftp.close()
            ssh.close()
            print("File downloaded successfully.")

            # Чтение содержимого файла в переменную
            with open(self.local_filepath, 'r') as local_file:
                file_content = local_file.read()

            return file_content
        except Exception as e:
            print(f"Failed to download file: {e}")
            return None

    def copy_file_to_remote(self, local_filepath):
        """Копирование файла для эмбеддинга на удаленный сервер"""

        # Укажите путь и имя файла на сервере
        # self.create_file("for_embedding001.txt")
        # local_filepath = "local_file.txt"
        remote_filepath = "/home/vl/r_files/" + local_filepath

        try:
            # Установление SSH соединения
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, self.port, self.username, self.password)

            # Использование SFTP для работы с файлами
            sftp = ssh.open_sftp()
            sftp.put(local_filepath, remote_filepath)

            sftp.close()
            ssh.close()
            print(f"File {local_filepath} copied to the server successfully.")

        except Exception as e:
            print(f"Failed to copy file: {e}")

    def copy_file_from_remote(self, local_filepath):
        """Копирование файла для эмбеддинга с удаленного сервера для того чтобы локально разбить его на чанки"""
        # Пока нет смысла использовать, так как файл и так в локальной директории
        # Укажите путь и имя файла на сервере
        remote_filepath = "/home/vl/r_files/" + local_filepath

        try:
            # Установление SSH соединения
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.hostname, self.port, self.username, self.password)

            # Использование SFTP для работы с файлами
            sftp = ssh.open_sftp()
            sftp.get(remote_filepath, local_filepath)

            sftp.close()
            ssh.close()
            print("File downloaded successfully.")

        except Exception as e:
            print(f"Failed to download file: {e}")

# Пример выполнения функций
#
# fh = ParamikoFiles()
# fh.copy_file_to_remote("rus_side_effects_embedding.txt")
#
# fh.write_to_file("padang padang")
# fh.create_and_save_file()
#
# content = fh.download_file()
#
# if content:
#     print("Content of the downloaded file:")
#     print(content)
