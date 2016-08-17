import operator
import argparse
# import progressbar
import re
import json

def getModalAnswer(answers):
	candidates = {}
	for i in range(10):
		candidates[answers[i]['answer']] = 1

	for i in range(10):
		candidates[answers[i]['answer']] += 1

	return max(candidates.items(), key=operator.itemgetter(1))[0]

def getAllAnswer(answers):
	answer_list = []
	for i in range(10):
		answer_list.append(answers[i]['answer'])

	return ';'.join(answer_list)

def counTokens(line):
    tokens = [word for word in re.split('\s+|[,.!?;"()]', line) if word.strip()]
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, default='train', 
        help='Specify which part of the dataset you want to dump to text. Your options are: train, val, test, test-dev')
    parser.add_argument('-answers', type=str, default='modal', 
        help='Specify if you want to dump just the most frequent answer for each questions (modal), or all the answers (all)')
    args = parser.parse_args()

    if args.split == 'train':
        annFile = '../data/coco_qa/answers/train/mscoco_train2014_annotations.json'
        quesFile = '../data/coco_qa/questions/train/OpenEnded_mscoco_train2014_questions.json'
        questions_file = open('../data/coco_qa/questions/train/questions_train2014.txt', 'w')
        questions_id_file = open('../data/coco_qa/questions/train/questions_id_train2014.txt', 'w')
        questions_lengths_file = open('../data/coco_qa/questions/train/questions_lengths_train2014.txt', 'w')
        if args.answers == 'modal':
            answers_file = open('../data/coco_qa/answers/train/answers_train2014_modal.txt', 'w')
        elif args.answers == 'all':
            answers_file = open('../data/coco_qa/answers/train/answers_train2014_all.txt', 'w')
        coco_image_id = open('./data/coco_qa/images/train/images_train2014.txt', 'w')
        data_split = 'training data'
    elif args.split == 'val':
        annFile = '../data/coco_qa/answers/val/mscoco_val2014_annotations.json'
        quesFile = '../data/coco_qa/questions/val/OpenEnded_mscoco_val2014_questions.json'
        questions_file = open('../data/coco_qa/questions/val/questions_val2014.txt', 'w')
        questions_id_file = open('../data/coco_qa/questions/val/questions_id_val2014.txt', 'w')
        questions_lengths_file = open('../data/coco_qa/questions/val/questions_lengths_val2014.txt', 'w')
        if args.answers == 'modal':
            answers_file = open('../data/coco_qa/answers/val/answers_val2014_modal.txt', 'w')
        elif args.answers == 'all':
            answers_file = open('../data/coco_qa/answers/val/answers_val2014_all.txt', 'w')
        coco_image_id = open('../data/coco_qa/images/val/images_val2014_all.txt', 'w')
        data_split = 'validation data'
    elif args.split == 'test-dev':
        quesFile = '../data/coco_qa/questions/test_dev/OpenEnded_mscoco_test-dev2015_questions.json'
        questions_file = open('../data/coco_qa/questions/test_dev/questions_test-dev2015.txt', 'w')
        questions_id_file = open('../data/coco_qa/questions/test_dev/questions_id_test-dev2015.txt', 'w')
        questions_lengths_file = open('../data/coco_qa/questions/test_dev/questions_lengths_test-dev2015.txt', 'w')
        coco_image_id = open('../data/coco_qa/images/test_dev/images_test-dev2015.txt', 'w')
        data_split = 'test-dev data'
    elif args.split == 'test':
        quesFile = '../data/coco_qa/questions/test/OpenEnded_mscoco_test2015_questions.json'
        questions_file = open('../data/coco_qa/questions/test/questions_test2015.txt', 'w')
        questions_id_file = open('../data/coco_qa/questions/test/questions_id_test2015.txt', 'w')
        questions_lengths_file = open('../data/coco_qa/questions/test/questions_lengths_test2015.txt', 'w')
        coco_image_id = open('../data/coco_qa/images/test/images_test2015.txt', 'w')
        data_split = 'test data'
    else:
        raise RuntimeError('Incorrect split. Your choices are:\ntrain\nval\ntest-dev\ntest')

    # initialize questions and answers
    questions = json.load(open(quesFile, 'r'))
    ques = questions['questions']
    print('number of questions: {0}'.format(len(ques)))
    if args.split == 'train' or args.split == 'val':
        qa = json.load(open(annFile, 'r'))
        ans = qa['annotations']
        # print('number of answers: {0}'.format(len(ans)))
    
    
    iterator = zip(range(len(ques)), ques)
    # pbar = progressbar.ProgressBar()
    print('Dumping questions, answers, questionIDs, imageIDs, and questions lengths to text files...')
    for i, q in iter(iterator):
        questions_file.write((q['question'] + '\n'))
        questions_id_file.write((str(q['question_id']) + '\n'))
        questions_lengths_file.write((str(len(counTokens(q['question']))) + '\n'))
        coco_image_id.write((str(q['image_id']) + '\n'))
        print('id: {0}, ques_id: {1}, question: {2}, ques_lengths: {3}, image_id: {4}'.format(
            i, q['question_id'], q['question'], len(counTokens(q['question'])), q['image_id']))
        if args.split == 'train' or args.split == 'val':
            if args.answers == 'modal':
                answers_file.write(getModalAnswer(ans[i]['answers']) + '\n')
            elif args.answers == 'all':
                answers_file.write(getAllAnswer(ans[i]['answers']) + '\n')
            # print('answers: {0}'.format(ans[i]['answers']))
  
    print('completed dumping ', data_split)

if __name__ == '__main__':
    main()
