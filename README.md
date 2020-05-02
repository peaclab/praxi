# Praxi
Praxi is a software discovery tool intended to assist software discovery in cloud systems. With cloud computing's increased prevalence in industry and research settings, monitoring the software present on large cloud servers is becoming more and more critical in ensuring their compliance, security, and efficiency. Praxi is an innovative method of software discovery combining the strengths of previous learning-based (DeltaSherlock) and practice-based (Columbus) techniques, resulting in a fast and accurate way to track applications installed or modified on cloud servers. 

See the repository wiki for a high level overview of Praxi.

Paper can be found here: https://www.bu.edu/peaclab/files/2020/03/PraxiJournal.pdf

## Requirements
* Linux
* Python 3.5+
* numpy
* scipy
* sklearn

## Installation

1. Ensure your system's up to date: `sudo apt update && sudo apt upgrade`
2. Install Vowpal Wabbit with the following two commands:
     
    `sudo apt-get install libboost-dev libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-math-dev libboost-test-dev zlib1g-dev cmake rapidjson-dev`
    
    `sudo apt-get install vowpal-wabbit`
2. Install PIP and ensure it's up to date: `sudo apt install python3-pip && sudo pip3 install --upgrade pip`
3. Install (or update) dependencies: `sudo pip3 install --upgrade watchdog numpy scipy sklearn tqdm envoy`
4. Clone this repo somewhere easy, like to your home directory: `git clone https://github.com/deltasherlock/praxi.git`

## Repository Organization
* **production_code**: contains scripts with command line flags for easy use (see wiki for use instructions)
* **research_code**: code used to generate results for IEEE paper.
* **cs_recorder**: code for recording changesets.
* **changeset_sets**: contains pickle files filled with changeset IDs. See README in folder for more information. 
* **demos**: contains code for IC2E and Middleware demos (see wiki for more information).
* **columbus**: Columbus code used to generate tagsets from changesets.

