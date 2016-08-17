import numpy as np

def concate(questionPath, answerPath, concationPath):
    questions = [q.strip() for q in open(questionPath).readlines()]
    answers = [a.strip() for a in open(answerPath).readlines()]

    i = 0
    with open(concationPath, 'wt') as writehandle:
        for q, a in zip(questions, answers):
            writehandle.write(q + ' ' + a + '\n')
            i += 1
    print('Concate finished! Total number: {0}.'.format(i))

if __name__ == '__main__':

    questionPath = '../data/coco_qa/questions/val/questions_val2014.txt'
    answerPath = '../data/coco_qa/answers/val/answers_val2014_modal.txt'
    concationPath = '../data/coco_qa/concateQA/concateqa_val.txt'

    concate(questionPath, answerPath, concationPath)