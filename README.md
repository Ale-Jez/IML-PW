# Introduction to Machine Learning

This is the repository for Introduction to Machine Learning course conducted at the Warsaw University of Technology during the winter semester 2025/26. 
It contains the project focused on voice analysis using machine learning methods.
The project was developed by a team of students:
- Aleksander Jeżowski
- Mantas Mikulskis
- Michał Kozicki
- Piotr Czechowski
- Rafał Lasota

This GitHub repository was created relatively lately, thus the commit history does not reflect actual workload,
which has been described in the section titled *Team Contributions* in the report for this project.

---

## Client App

The repository includes a functional client application located in the `client` folder. This application demonstrates the practical performance of our trained models and operates in two distinct modes:

* **Binary Classification:** Determines whether a voice sample belongs to a specific category using our binary classification model.
* **Speaker Identification:** Analyzes the voice input to identify the specific speaker from a known set of individuals using our multi-class identification model.

The app serves as a visualization of the inference results of the machine learning pipeline developed in this project.

## Installation and Usage

Firstly, launch the server.

```bash
cd backend
```

If this is the first time launching, create a venv/conda and
```bash
pip install -r requirements.txt
```

Then
```bash
uvicorn main:app --host localhost --port 8000                                                                                                ─╯
```

Then, in another terminal window launch the client.
```bash
cd frontend
```
If this is the first time launching,
```bash
npm install
```

Then
```bash
npm run dev
```

## Report

A comprehensive report detailing our methodology, architecture choices, experimental results, and team contributions is available as a pdf file in the `REPORT` folder.

## Project Structure

In addition to the final report and the client application, this repository contains the complete source code used throughout the project lifecycle, mainly files in `.py` and `.ipynb` formats. Mainly including:

- **Data Preprocessing:** Scripts for cleaning and normalizing audio data.
- **Feature Extraction:** Code used to generate spectrograms, MFCCs, or other voice features.
- **Model Training:** Notebooks and scripts used to train, tune, and evaluate the models.



