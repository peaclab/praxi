#!/usr/bin/python3

import sys
sys.path.insert(0, '../')
from cs_recorder import  changesets, ds_watchdog, io
import os
import json
import yaml

#cs = changesets.Changeset(open_time = time())
def json_to_yaml(fname, yamlname, label=None):
    with open(fname) as json_file:
        data = json.load(json_file)

    changes = set()
    open_time = data['open_time']
    close_time = data['close_time']

    for f_create in data['creations']:
        changes.add(f_create['filename'])

    for f_create in data['modifications']:
        changes.add(f_create['filename'])

    for f_create in data['deletions']:
        changes.add(f_create['filename'])

    changes = list(changes)

    # create dictionary and save to yaml file
    yaml_in = {'open_time': open_time, 'close_time': close_time, 'label': label, 'changes': changes}
    with open(yamlname, 'w') as outfile:
        yaml.dump(yaml_in, outfile, default_flow_style=False)


if __name__ == '__main__':
    watch_paths = ["~/praxi"] # ["/var/", "/bin/", "/usr/", "/etc/"]
    dswd = ds_watchdog.DeltaSherlockWatchdog(watch_paths, "*", ".")
    # Recording begins immediately after instantiation.
    print("Recording started")
    input("Press Enter to continue...")
    print("Recording stopped")

    # Save changeset
    cs = dswd.mark()
    print("Saving as json")
    io.save_object_as_json(cs, "cs.dscs")

    label = input("Input the label for this changeset:")

    yamlname = input("Input the filename for this changeset (ending in .yaml):")

    json_to_yaml('cs.dscs', yamlname, label=label)

    # Remove json file
    os.remove("cs.dscs")

    print("Done!")
