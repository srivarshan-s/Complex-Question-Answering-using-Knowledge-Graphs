from datetime import datetime
import pandas
from flask import Flask, jsonify, redirect, render_template, request, url_for
from QnA import *


app = Flask(__name__)


class CheckAndSave:

    def __init__(self):
        super(CheckAndSave, self).__init__()

    def createdataset(self, para, que, ent, ans1, ans2):
        wholedata = {"para":[str(para)],"que":[[str(que)]], "entities":[ent], "ans1": [ans1], "ans2":[ans2]}


class QnaModel:
    def __init__(self):
        self.getent = ExtractEntity()
        self.qa = QuestionAnswer()
        self.export = exportJson()

    def getAnswer(self, paragraph, question):

        refined_text = self.getent.preprocess_text([paragraph])
        dataEntities, numberOfPairs = self.getent.get_entity(refined_text)

        if dataEntities:
            # data_in_dict = dataEntities[0].to_dict()
            self.export.export_data(dataEntities[0])
            outputAnswer = self.qa.find_answer(str(question), numberOfPairs)
            if outputAnswer == []:
                return None
            return outputAnswer
        return None


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route('/clear', methods=['GET', 'POST'])
def clear():
    return redirect(url_for('main'))


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    model = QnaModel()
    save = CheckAndSave()
    input_paragraph = str(request.form["paragraph"])
    input_question = str(request.form["question"])
    my_answer = model.getAnswer(input_paragraph, input_question)

    return render_template('index.html', my_answer=my_answer, input_paragraph=input_paragraph ,input_question=input_question)

if __name__ == "__main__":
    app.run(host="localhost", port=8000, threaded=True)
