# URL : http://casy.cse.sc.edu/kite

# KITE

## Code
* [helpers.py](vis/Helpers/helpers.py) : Contains code for training models and pre-processing
* [helpers2.py](vis/Helpers/helpers2.py) : Contains actual code to generate a visualization
* [views.py](vis/views.py) : Compile results from [helpers2.py](vis/Helpers/helpers2.py)
* [rough.ipynb](vis/Helpers/rough.ipynb) : Rough experimentation with code is done here

## Data
[Data](vis/Data)

## Models
[Trained embeddings of all domains](vis/Models)

## HTML
[HTML Files](vis/templates/vis)

## Evaluation Data
[CSV files to be annotated](Evaluation%20Data) (Incomplete right now)

## Requirements
[Python requirements](requirements.txt)

## Installation Instructions
* Install Python3
* Open the Command Prompt/Terminal
* Install requirements
```bash
pip install -r requirements.txt
```
OR
```bash
pip3 install -r requirements.txt
```
(Whichever works)
* Install Spacy dependency

For Linux/MacOs
```bash
python3 -m spacy download en_core_web_lg
python3 -m spacy download en_core_web_sm
```
For Windows
```bash
py -m spacy download en_core_web_lg
py -m spacy download en_core_web_sm
```
* Change directory to this Django project
* Apply migrations (Only for the first time when installing)  

For Linux/MacOS
```bash
python3 manage.py migrate
```
For Windows
```bash
py -m manage.py migrate
```
* Run  

For Linux/MacOS
```bash
python3 manage.py runserver
```
For Windows
```bash
py -m manage.py runserver
```
* Visit the link shown on the terminal to view the web app.

## Links for installation
* [Python installation](https://www.python.org/downloads/)
