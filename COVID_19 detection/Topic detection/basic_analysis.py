import nltk
from collections import Counter
stopwords_set = set(nltk.corpus.stopwords.words('english'))

stopwords_set.update(("retweet","rt","rts","women","men","thank","thanks","tweet","tweets","let","get","got","think","vote","lot","lots","say","says","know","guys","guy","today","lol",\
	"tonight","week","people","true","make","need","good","day","win","look","looks","really","thing","said","say","says","tell","told","tells","days","want","voting","making",\
	"way","man","news","come","twitter","yes","real","big","best","girl","right","person","dont","wish","right","poll","video","hope","life","soon","wait","gon","na","going","fav",\
	"ago","years","make","covid","coronavirus"))

subsList = {}


filter_set = stopwords_set

# Topic keywords
Topics = list()
Topics.append(['sipping', 'water', 'every', 'minutes', 'voice', 'prevent', 'hot'])
Topics.append(['breath', 'hold', 'seconds', 'holding', 'fact', 'check', 'disease'])
Topics.append(['hot', 'bath', 'tacking', 'prevent', 'new', 'disease', 'water'])
Topics.append(['mosquito', 'bites', 'govt', 'kill', 'like', 'transmitted', 'virus'])
Topics.append(['antibiotics', 'virus', 'work', 'bacteria', 'infection', 'cure', 'treat'])
Topics.append(['water', 'salt', 'hot', 'kill', 'colloidal', 'used', 'warm'])
Topics.append(['water', 'garlic', 'hot', 'throat', 'dry', 'drink', 'virus'])
Topics.append(['weather', 'cold', 'flu', 'hot', 'virus', 'move', 'much'])
Topics.append(['transmitted', 'hot', 'humid', 'areas', 'climate', 'virus', 'cannot'])
TopicsNames = ['sipping water','hold breathe','hot bath','mosquiton','antibiotics','salt water','garlic water','cold weather','hot and humid weather']


def wordFilter(wordList, filterWords):
    return [word for word in wordList if word not in filterWords]


def basic_analysis(normalWordbag):
    normalWordbag = wordFilter(normalWordbag, filter_set)
    print('%d non-stop words totally.' % len(normalWordbag))
    normalCounter = Counter(normalWordbag)
    print('%d non-repeative words.' % len(normalCounter))
    print("Most 30 common words:")
    print(normalCounter.most_common(30))

    return

