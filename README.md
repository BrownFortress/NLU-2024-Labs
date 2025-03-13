# NLU-2024 Laboratory

This repo contains all the laboratory lectures for the Natural Language Understanding course held at the University of Trento.
<br>

## Getting started

We suggest you install [Anaconda](https://www.anaconda.com/download) on your laptop and import the conda environment that we have prepared for you. The reason for this is to give you the same library versions that we used to test the labs. However, if you are not a conda lover, you can manually install on your favourite virtual env the libraries listed in the `requirements.txt` file.

```bash
conda env create -f nlu_env.yaml -n nlu24
conda activate nlu24
```
If you have a Mac or a Windows or you do not have a dedicated Nvidia gpu, you can install the environment in this way:

```bash
conda create -n nlu24 python=3.10.13
conda activate nlu24
pip install -r requirements_no_cuda.txt
```

To launch a lab run this line of code:
```bash
jupyter notebook
```


Then, you have to choose the lab that you want to open. 

<br>

## Repo organization
In the repo `labs`, you can find the notebooks for each lab session and in `solutions` you can find the same notebooks with the solutions.
<br>

The solutions of each lab will be uploaded after the corresponding lab lecture.


## Exam

### Instructions
The mandatory projects are presented in lab 4 (LM) and 5 (NLU), while the optional project is presented in lab 6 (SA). The projects LM and NLU will give you 15/30 points each. SA is to reach the top grade i.e. 30 lode. 

For each of the projects, you must write a small report following the LaTeX template in the zip folder `report_template.zip`. In particular, you have to write a mini-report of **max 1 page** (references, tables and images are excluded from the count) in which you explain all the parts of the project giving more weight to the part with higher points. **Reports longer than 1 page will not be evaluated**. The purpose of this is to give you a way to report **cleanly** the results and give you space to describe what you have done and/or the originality that you have added to the exercise. You can find more detail about the sections and relative content in the LaTeX template.

### Grading
For each project, the final grade is based on:
- Code review;
- Report review;
- Q&A at the exam.
    -  The questions will be related to the delivered projects, the associated models/solutions and concepts supporting the theory.


### Submission format

The delivery must follow the directory schema that you can find in `exam/studentID_name_surname.zip`. 

The `LM` and `NLU` folders contain two sub-folders one for part 1 and the other for part 2, while for SA there is only part 1.   Inside them, there are the following files and folders: `main.py`, `functions.py` ,  `utils.py`,  `model.py`,  `README.md`, `/dataset` and `/bin`.

- `utils.py`: you have to put all the functions needed to preprocess and load the dataset
- `model.py`: the class of the model defined in PyTorch.
- `functions.py`: you have to write all the other required functions (taken from the notebook and/or written on your own) needed to complete the exercise.
- `main.py`: you have to write the calls to the functions needed to output the results asked by the exercise.
- `README.md`: you may want to write a message for us related to your solution (optional).
- `/dataset`: the files of that dataset that you used.
- `/bin`: the binary files of the best models that you have trained.

 The **reports** have to be placed in the corresponding projects folders i.e. into the folders `LM`, `NLU` and `SA`.

**Last but not least**, the code has to be **well-written** and **documented** with comments. Furthermore, the script has to **run** without bugs otherwise the exercise will not be evaluated. Jupyter notebooks are not accepted.

<br>

### How to submit
To submit your work you have to fill out this [Google form](https://forms.gle/2UNFNRDDAboTgeuG9). The work must be delivered **7 days before** the date of the session exam that you want to attend. You can do multiple submissions as we will check only the last one.

<br>

## Acknowledgements
The notebooks that you can find here are an adaptation of the labs created by our colleague [Evgeny A. Stepanov](https://github.com/esrel).
