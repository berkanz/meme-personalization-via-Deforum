import os
import numpy as np
import shutil
    
def cleanup_after_render(path):
    [os.remove(os.path.join(path, file)) for file in os.listdir(path) if file.endswith('.png')]
    [os.remove(os.path.join(path, file)) for file in os.listdir(path) if file.endswith('.txt')]
    if os.path.exists(os.path.join(path,"inputframes")):
        shutil.rmtree(os.path.join(path,"inputframes"))
    return