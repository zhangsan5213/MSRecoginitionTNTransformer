import os
from tqdm import tqdm

docker_names = ["xenodochial_haibt", "dazzling_wilson", "vigilant_hamilton", "sleepy_darwin",
                "dreamy_morse", "quirky_kowalevski", "inspiring_galois", "admiring_lalande",
                "busy_vaughan", "musing_cerf", "trusting_turing", "practical_brahmagupta",
                "jovial_curie", "dazzling_curie", "exciting_heyrovsky", "gallant_borg",
                "recursing_beaver", "sleepy_hamilton", "sharp_newton", "magical_dijkstra",
                "lucid_goldwasser", "ecstatic_volhard", "peaceful_bhabha", "competent_wilbur",
                "eloquent_cray", "zen_elbakyan", "vigilant_dirac", "blissful_williams",
                "bold_sutherland", "ecstatic_bhabha", "inspiring_fermi", "focused_lamarr",
                "pensive_solomon", "wonderful_hypatia", "festive_mccarthy", "epic_heisenberg",
                "vibrant_wing", "peaceful_cartwright", "magical_shaw", "funny_brahmagupta"]

tolerance = "0.001"
output_file_path = "../trained_models_cfmid4.0/[M+H]+/param_output.log"
config_file_path = "../trained_models_cfmid4.0/[M+H]+/param_config.txt"

smi_agent_path  = "/home/lrl/dataset/zinc_smi/agent/"
smi_agent_paths = [smi_agent_path + i for i in os.listdir(smi_agent_path)]
smi_stock_path  = "/home/lrl/dataset/zinc_smi/in_stock/"
smi_stock_paths = [smi_stock_path + i for i in os.listdir(smi_stock_path)]
smi_paths = smi_agent_paths + smi_stock_paths

commands = []
index = 0
for smi in tqdm(smi_paths):
    if not smi.endswith("smi"): continue
    lines = open(smi).readlines()[1:]
    for line in lines:
        try:
            smiles = line.split()[0]
            temp = "docker exec {} cfm-predict \"{}\" {} {} {} 1 {}.txt 1 0".format(
                docker_names[index%(len(docker_names))],
                smiles, tolerance, output_file_path, config_file_path, str(index).rjust(8, "0"))
            commands.append(temp)
            index += 1
        except:
            continue

import subprocess
import datetime
import os
import threading

def execCmd(cmd):
    try:
        os.system(cmd)
    except:
        print('%s\t FAILED!' % (cmd))

for i in tqdm(range(int(len(commands)/len(docker_names)))):
    threads = []
    for j in range(len(docker_names)):
        th = threading.Thread(target=execCmd, args=(commands[len(docker_names)*i + j],))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()