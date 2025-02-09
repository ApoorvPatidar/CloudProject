<<<<<<< HEAD
## Simple purpose of this file is to keet the log of all the things that are happening in the application.

import logging
import os 
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
=======
## Simple purpose of this file is to keet the log of all the things that are happening in the application.

import logging
import os 
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
>>>>>>> 3c5f526f04bc47a82dd07db8730d77a390371e9c
os.makedirs(logs_path, exist_ok=True)