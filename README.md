## Yektanet website classification
<p dir=ltr style="direction: ltr; text-align: justify; line-height:200%; font-family:vazir; font-size:medium">
<font face="vazir" size=3>
The project involves using real data from Persian websites, which has been refined and collected by the <a href="https://www.yektanet.com">Yektanet</a> platform. The goal is to build a machine learning model that can predict the category of a website based on its content, such as the title, description, and full text. For instance, if we have a link from a news site with the title "کیهان کلهر جایزه مرد سال موسیقی جهان را دریافت کرد", our model should be able to predict that this article is related to the topic of "music". We can use not only the title but also the description or full text to make predictions.
</font>
</p>

![image](https://github.com/user-attachments/assets/5e77b758-79cf-4bf4-95c4-9502eca03e0f) 

<h2 align=left style="line-height:200%;font-family:vazir;color:#0099cc">
<font face="vazir" color="#0099cc">
Introduction to the dataset
</font>
</h2>

<p dir=ltr style="direction: ltr; text-align: justify; line-height:200%; font-family:vazir; font-size:medium">
<font face="vazir" size=3>
Each instance of this dataset is associated with the attributes described in the table below. column<code>category</code> is the target variable of the issue that represents the subject of the content.
</font>
</p>
<center>
<div dir=ltr style="direction: ltr;line-height:200%;font-family:vazir;font-size:medium">
<font face="vazir" size=3>
    
|columns|description|
|:------:|:---:|
|<code>category</code>| subjects (target variable) |
|<code>description</code>| description |
|<code>text_content</code>| text contents |
|<code>title</code>| title of the website |
|<code>h1</code>| content of <code>h1</code> tag in page |
|<code>h2</code>|content of <code>h2</code> tag in page  |
|<code>url</code>| address of webpage|
|<code>domain</code>|website domain |
|<code>id</code>| website id|

</font>
</div>
</center>


## What is Tfidftransformer and difference between Tfidftransformer and Tfidfvectorizer
### Tfidftransformer vs. Tfidfvectorizer
In summary, the main differences between the two modules are as follows:
+ With Tfidftransformer you will systematically compute word counts using CountVectorizer and then compute the Inverse Document Frequency (IDF) values and only then compute the Tf-idf scores.
+ With Tfidfvectorizer on the contrary, you will do all three steps at once. Under the hood, it computes the word counts, IDF values, and Tf-idf scores all using the same dataset.

### When to use what?
So now you may be wondering, why you should use more steps than necessary if you can get everything done in two steps. Well, there are cases where you want to use Tfidftransformer over Tfidfvectorizer and it is sometimes not that obvious. Here is a general guideline:

+ If you need the term frequency (term count) vectors for different tasks, use Tfidftransformer.
+ If you need to compute tf-idf scores on documents within your “training” dataset, use Tfidfvectorizer
+ If you need to compute tf-idf scores on documents outside your “training” dataset, use either one, both will work.

## What is SMOTE

A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary.

One way to solve this problem is to oversample the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset before fitting a model. This can balance the class distribution but does not provide any additional information to the model.

An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be very effective.

Perhaps the most widely used approach to synthesizing new examples is called the **Synthetic Minority Oversampling TEchnique**, or **SMOTE** for short. This technique was described by *Nitesh Chawla*, et al. in their 2002 paper named for the technique titled *“SMOTE: Synthetic Minority Over-sampling Technique.”*

SMOTE works by selecting examples that are close to the feature space, drawing a line between the examples in the feature space, and drawing a new sample at a point along that line.

Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found *(typically k=5)*. A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.

<img src="https://github.com/user-attachments/assets/2e493da2-c019-4b1f-961b-f3920928bcf5" alt="image" width="300" />
<img src="https://github.com/user-attachments/assets/ae87c2a0-4b5b-44e5-bfa1-a5bd5137e9d1" alt="image" width="300" />

*Scatter Plot of Imbalanced Dataset Transformed by SMOTE and Random Undersampling*

## Results
| Model                    | Accuracy  | Precision  | Recall    | F1-Score  |
|--------------------------|-----------|------------|-----------|-----------|
| LogisticRegression       | 0.944115  | 0.946159   | 0.944115  | 0.944383  |
| LinearSVC                | 0.967802  | 0.967932   | 0.967802  | 0.967645  |
| DecisionTreeClassifier    | 0.855662  | 0.855326   | 0.855662  | 0.854731  |
| RandomForestClassifier    | 0.935603  | 0.937822   | 0.935603  | 0.935868  |
| KNeighborsClassifier      | 0.878979  | 0.889758   | 0.878979  | 0.872444  |

## Resource
+ [How to Use Tfidftransformer & Tfidfvectorizer?](https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#:~:text=With%20Tfidftransformer%20you%20will%20systematically,all%20three%20steps%20at%20once.)
+ [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
