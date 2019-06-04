# Run tutorial quickly
import envoy
import subprocess

subprocess.call(["./demo_main.py", "-t", "demo_tagsets/sl_train_tag", "-o",  "results",  "-n", "4"]) # Run command
subprocess.call(["./demo_main.py", "-t", "demo_tagsets/sl_train_tag", "-s", "demo_tagsets/sl_test_tag", "-o", "results"]) # Run command
subprocess.call(["./demo_main.py", "-t", "demo_tagsets/ml_train_tag", "-o", "results", "-m", "-n", "4"]) # Run command
subprocess.call(["./demo_main.py", "-t", "demo_tagsets/ml_train_tag", "-s", "demo_tagsets/ml_test_tag", "-o", "results", "-m"]) # Run command
subprocess.call(["./demo_main.py", "-t", "demo_tagsets/iter_init", "-s", "demo_tagsets/sl_test_tag", "-o", "results", "-i", "it_model.vw", "-l"]) # Run command
subprocess.call(["./demo_main.py", "-t", "demo_tagsets/iter_add", "-s", "demo_tagsets/sl_test_tag", "-o", "results", "-i", "it_model_2.vw", "-p", "it_model.p"]) # Run command
print("Done!")
