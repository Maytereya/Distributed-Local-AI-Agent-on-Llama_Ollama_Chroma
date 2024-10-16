import configparser

config = configparser.ConfigParser()
config.read('config.ini')

environment = 'DEVELOPMENT' # OR 'PRODUCTION'

ollama_url = config[environment]['ollama_url']
chroma_host = config[environment]['chroma_host']
chroma_port = int(config[environment]['chroma_port'])

emb_model = config['DEFAULT']['emb_model']
ll_model_big = config['DEFAULT']['ll_model_big']
ll_model = config['DEFAULT']['ll_model']
question1 = config['DEFAULT']['question1']
collect_name = config['DEFAULT']['collect_name']
