import configparser

config = configparser.ConfigParser()
config.read('config.ini')

environment = 'PRODUCTION'  # OR 'DEVELOPMENT'

ollama_url = config[environment]['ollama_url']
chroma_host = config[environment]['chroma_host']
chroma_port = int(config[environment]['chroma_port'])

emb_model = config['DEFAULT']['emb_model']
ll_model = config['DEFAULT']['ll_model']
question1 = config['DEFAULT']['question1']
