import pandas as pd

def get_data(directory, dataset="filename",n_component = 2):

    data = pd.read_csv(directory + '/' + dataset +"_"+ str(n_component) +".csv",index_col=0)
    #Get clustered data based on labels

    cluster_1 = data[data.iloc[:,0].values==1.0]
    cluster_2 = data[data.iloc[:,0].values==2.0]
    #cluster_3 = data[data.iloc[:,0].values==3.0]
    #cluster_4 = data[data.iloc[:,0].values==4.0]

    #Data preprocessing
    sum_cluster_1 = cluster_1.iloc[:,1:].sum(axis=0)
    sum_cluster_2 = cluster_2.iloc[:,1:].sum(axis=0)
    #sum_cluster_3 = cluster_3.iloc[:,1:].sum(axis=0)
    #sum_cluster_4 = cluster_4.iloc[:,1:].sum(axis=0)
    sum_data = data.iloc[:,1:].sum(axis=0)

    train_macro = sum_data[:800]
    test_macro = sum_data[800:]

    train_micro_1 = sum_cluster_1[:800]
    test_micro_1 = sum_cluster_1[800:]

    train_micro_2 = sum_cluster_2[:800]
    test_micro_2 = sum_cluster_2[800:]

    #train_micro_3 = sum_cluster_3[:800]
    #test_micro_3 = sum_cluster_3[800:]

    #train_micro_4 = sum_cluster_4[:800]
    #test_micro_4 = sum_cluster_4[800:]

    return train_macro, test_macro, train_micro_1, test_micro_1, train_micro_2,test_micro_2#, train_micro_3, test_micro_3#, train_micro_4, test_micro_4
