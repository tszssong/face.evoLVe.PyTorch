import yaml
import pickle

f = open('w32_256x192_adam_lr1e-3.yaml')
content = yaml.load(f)
with open('w32_256x192_adam_lr1e-3.pkl','wb') as file:
    pickle.dump(content,file)