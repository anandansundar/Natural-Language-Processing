import wikipedia
import csv
from nltk.metrics import *

reader = csv.reader(open('Topic_set_all.csv', 'r'))

actual_class = []
obtained_class = []

for row in reader:
    print(row)
    title, category = row
    wiki_page = wikipedia.page(title)
    wiki_content = str.lower(wiki_page.content)
    tech_count = wiki_content.count("technology")
    politics_count = wiki_content.count("politics")
    business_count = wiki_content.count("business")
    travel_count = wiki_content.count("travel")
    max_count = max(tech_count, politics_count, business_count, travel_count)

    class_atr = "politics"

    if max_count == travel_count:
        class_atr = "travel"
    if max_count == tech_count:
        class_atr = "technology"
    if max_count == business_count:
        class_atr = "business"

    print("Actual Class : " +category)
    print("Obtained Class : " +class_atr)
    actual_class.append(category.strip())
    obtained_class.append(class_atr)

accuracy_baseline = accuracy(obtained_class, actual_class) * 100

print('Accuracy of baseline : ' + accuracy_baseline.__str__() + "%")