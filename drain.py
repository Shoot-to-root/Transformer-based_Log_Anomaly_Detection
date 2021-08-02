import json
import logging
import os
import subprocess
import sys
import time
from os.path import dirname
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

#in_log_file = "./HDFS/HDFS.log"
#in_log_file = "./BGL/BGL.log"
#in_log_file = "./openstack/openstack_normal1.log"
#in_log_file = "./openstack/openstack_normal2.log"
in_log_file = "./openstack/openstack_abnormal.log"

config = TemplateMinerConfig()
config.load(dirname(__file__) + "/drain3.ini")
config.profiling_enabled = False
template_miner = TemplateMiner(config=config)

line_count = 0

with open(in_log_file) as f:
    #lines = [next(f) for x in range(5)]
    lines = f.readlines()
"""  
for line in lines:
    #print(line)
    line = line.rstrip()
    line = line.partition(": ")[2]
    print(line)
  
"""  
templates = []
parameters = []

start_time = time.time()
batch_start_time = start_time
batch_size = 10000

for line in lines:
    line = line.rstrip()
    line = line.partition(": ")[2]
    
    if "INFO" in line:
        line = line.partition("INFO ")[2]
    if "ERROR" in line:
        line = line.partition("ERROR ")[2]
    if "WARNING" in line:
        line = line.partition("WARNING ")[2]
    if "CRITICAL" in line:
        line = line.partition("CRITICAL ")[2]
        
    result = template_miner.add_log_message(line)
    line_count += 1
    if line_count % batch_size == 0:
        time_took = time.time() - batch_start_time
        rate = batch_size / time_took
        batch_start_time = time.time()
        
    if result["change_type"] != "none":
        result_json = json.dumps(result)
        logger.info(f"Input ({line_count}): " + line)
        logger.info("Result: " + result_json)
        template = result["template_mined"]
        templates.append(template)
        param = template_miner.get_parameter_list(template, line)
        parameters.append(param)
        logger.info("Parameters: " + str(param) + "\n")

df = pd.DataFrame({"Template": templates,
                       "Parameter": parameters})
df.to_csv("./openstack/openstack_abnormal.csv", index=False)

time_took = time.time() - start_time
rate = line_count / time_took
logger.info(f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.drain.clusters)} clusters")
