import pandas as pd
from yellowbrick.classifier import ConfusionMatrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



#visualização da matriz da confusão
def visual(x_train,y_train,x_test,y_test):    
    v=ConfusionMatrix(GaussianNB())
    v.fit(x_train,y_train)
    v.score(x_test,y_test)
    v.poof()

#print valores accuray etc
def printValues(tn,fp,fn,tp):
    precision = ((tp)/(fp+tp))
    recall = ((tp)/(fn+tp))
    accuracy = ((tp + tn)/(tn + fp + tp + fn ))
    f1 = (2*(precision*recall)/(precision + recall))
    errorRate=1-accuracy
    print('Accuracy Score: %.3f' % accuracy)
    print('Precision Score: %.3f' % precision)
    print('Recall Score: %.3f' % recall)
    print('Error rate: %.3f '% errorRate)
    print('F1 Score: %.3f' % f1)


#analisar moda media e mediana
def analysis(lista,df):
    for k in lista:
        if k != 'Unnamed: 0' and k != 'match':
            print("Média",k,"=",df[k].mean())
            print("Mediana",k,"=",df[k].median())
            print("Mode",k,":",df[k].mode())
            print("\n")

def main():
	
    pd.options.display.width = 0
    df = pd.read_csv("speedDating_trab.csv")
    #total de instancias
    total= 8378
    countNan=0
    
    print("Tamanho inicial do Dataset ",len(df.index))
    print("\nColunas: nº de NA:")
    print(df.isnull().sum(axis = 0))
    
    print("\nColunas: percentagem de NA")
    print(df.isnull().sum(axis = 0)/total *100)
   
    print("\nApagar as colunas: length")
    #Apagar colunas
    del df['length']
    
    #eliminar instancias com 2 ou mais (Na)
    print("\nEliminar as linhas com 2 ou mais NA")
    for x in range(total):
      if(df.loc[[x]].isna().sum().sum() >= 2):
       df=df.drop(x) 
       countNan+=1
       
    print("\nForam eliminadas ",countNan, " linhas do dataset") 
    print("\nTamanho atual:",len(df.index))
    #print(df.isnull().sum())
    per=countNan/total*100
    print("\nEliminamos ", per ,"% do dataset\n\n") 
    print(df.isnull().sum(axis = 0))
    df=df.reset_index(drop=True)
    lista = list(df)
    print("\nAnalisar a tabela")
    analysis(lista,df)
    
    
    opc= input("Substituir os NAs pela média (pressiona 1) ou pela mediana (pressiona 2) ou pela moda( pressiona 3) \n\n")    
    
    if opc == "1":
     #media
     df=df.fillna(df.mean())
    
    if opc == "2":
     #mediana
     df=df.fillna(df.median())
    
    if opc == "3":
     #moda
     df=df.fillna(df.median())
      
   
    
    #Correlação
    td=df[["id","partner","age","age_o","goal","date","go_out","int_corr","met","like","prob","match"]]
    #import matplotlib.pyplot as plt
    #plt.matshow(td.corr())
    #plt.show()
    
    
    #f_classif: é adequado quando os dados são numéricos e a variável alvo é categórica. 
    #k=int(input("Metodo f_classif escreva um número para o k\n\n"))
    
    a=df[["id","partner","age","age_o","goal","date","go_out","int_corr","met","like","prob"]]
    b=df["match"]
    #from sklearn.feature_selection import SelectKBest
    #from sklearn.feature_selection import f_classif, mutual_info_classif
    #f_classif = SelectKBest(score_func=f_classif, k=k)
    #fit = f_classif.fit(a,b)
    #features = fit.transform(a)

    #visualizar as features
    #cols = fit.get_support(indices=True)
    #print(a.iloc[:,cols])
    
    ntest= input("Quantos atributos? : todos(pressiona 1),3(pressiona 2),4(pressiona 3),6(pressiona 4),7(pressiona 5),8(pressiona 6),9(pressiona 7) \n\n") 
    
    #testar com todos atributos
    if ntest=="1":
     x = df[["id","partner","age","age_o","goal","date","go_out","int_corr","met","like","prob"]]
     y = df["match"]
    #testar com 3 atributos
    if ntest=="2":
     x = df[["date","like","prob",]]
     y = df["match"]
    #testar com 4 atributos
    if ntest=="3":
     x = df[["date","go_out","like","prob"]]
     y = df["match"]
    #testar com 6 atributos
    if ntest=="4":
     x = df[["age_o","date","go_out","met","like","prob"]]
     y = df["match"]
    #testar com 7 atributos
    if ntest=="5":
     x = df[["age","age_o","date","go_out","met","like","prob"]]
     y = df["match"]
    #testar com 8 atributos
    if ntest=="6":
     x = df[["age","age_o","date","go_out","int_corr","met","like","prob"]]
     y = df["match"]
    #testar com 9 atributos
    if ntest=="7":
     x = df[["age","age_o","goal","date","go_out","int_corr","met","like","prob"]]
     y = df["match"]
    
    opc1= input("Pressiona 1 para holdout ou 2 para cross validation \n\n")    
    
    naive_bayes=GaussianNB()
    
    if opc1== "1": 
    
     #divisao da base de dados entre treino e testes
     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    
     print("Número de testes")
     print(len(x_test.index))
     
     #informação do teste e treino
     print("\nInformações do teste")
     counts = y_test.value_counts().to_dict()
     print(counts)
     
     print("\nInformações do treino")
     countstrain = y_train.value_counts().to_dict()
     print(countstrain)
     
     #criação e treinamento do modelo
     naive_bayes.fit(x_train,y_train)
    
     #previsão utilizado testes
     pred=naive_bayes.predict(x_test)
     print("\n")
    
     #matriz de confusão e calculo da taxa de acerto e erro
     conf=confusion_matrix(y_test,pred)
     print("Matriz de confusão\n")
     print(conf)
     print("\n")
     accuracy=accuracy_score(y_test,pred)
     errorRate=1-accuracy
     
    
     #outros valores
     print('Accuracy score: %.3f' % accuracy)
     print('Error rate: %.3f '% errorRate)
     print('Precision Score: %.3f' % precision_score(y_test,pred))
     print('Recall Score: %.3f' % recall_score(y_test,pred))
     print('F1 Score: %.3f' % f1_score(y_test,pred))
    
     #grafico
     visual(x_train,y_train,x_test,y_test)
    
    
    if opc1=="2":
      count_tn = 0
      count_fp = 0
      count_fn = 0
      count_tp = 0
      kfold = KFold(n_splits=5,random_state=42, shuffle=True)
      for train_index, test_index in kfold.split(x):
       x_train , x_test = x.iloc[train_index,:],x.iloc[test_index,:]
       y_train , y_test = y[train_index] , y[test_index]
       naive_bayes.fit(x_train,y_train)
       pred = naive_bayes.predict(x_test)
       
       #informação do teste e treino
       print("\nInformações do teste")
       counts = y_test.value_counts().to_dict()
       print(counts)
     
       print("\nInformações do treino")
       countstrain = y_train.value_counts().to_dict()
       print(countstrain)
     
       visual(x_train,y_train,x_test,y_test)
       tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
       count_tn += tn
       count_fp += fp
       count_fn += fn
       count_tp += tp
	
      print("\n")
      printValues(count_tn,count_fp,count_fn,count_tp)



if __name__ == "__main__":
    main()















