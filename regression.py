"""
Windows
ctrl+c
conda activate pytorchenv | python regression.py | tee regressionlog1.txt
conda activate pytorchenv | python regression.py > regressionlog1.txt
conda init pytorchenv | python regression.py | tee regressionlog1.txt
conda init pytorchenv | python regression.py > regressionlog1.txt

Linux
python3 regression.py | tee regressionlog1.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
import os # for hadling creation of foders
import time # for meassuaring time
#
#
pd.set_option('display.max_columns',100)
pd.set_option('display.min_rows',10)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width',200)
pd.set_option('expand_frame_repr',100)
pd.set_option('display.max_colwidth',100)
#
#
print("Libraries Imported")
#
#
#  lets create a class for a stadard model of neural network
class standardClassificationModel101(nn.Module): # Methods and attributes are inherited from nn.Module
    instances = [] # lits to track the number of instances of this class
    counter = 0

    def __init__(self,in_Features=4,H1=8,H2=9,out_Features=3):
        # self: refers to the stantiation of the current object itself
        # in_Features: input Features, we are ussing default values (
        #           wind speed, 
        #           wind direction, 
        #           draft, 
        #           maximum tension from previous hours, 
        #           minimum tension from previous hours, 
        #           tension gradients???
        #       )
        # H1 : number of neurons of the hidden layer #1
        # H2 : number of neurons of the hidden layer #2
        # out_Features : number of categories of the output values 
        #
        # we need to define how many layers we need        
        # for classifying iris flowers we need 4 input features and the model will have 2 hidden layers
        # with N neurons each and an output layer with 3 nodes, one for each class
        #   4f  -- > h1 N -- > h2 N -- > 3o (3classes)
        #
        #debug#print(f'An NN object of class standardClassificationModel101 was created ')
        standardClassificationModel101.instances.append(self) # adding a new instance to the list
        standardClassificationModel101.counter += 1
        #debug#print(f'Instances created so far: #{standardClassificationModel101.counter}')
        #
        # super(standardClassificationModel101,self).__init__()
        super().__init__() # THis line stantiates the parent class from where we are inhering the class constructor, attributes and methods
        # The supper call delegates the function call to the parent class, which is nn.Module. This is needed to initialize the
        # nn.Module properly. It returns a temporary object of the superclass that then allows you to call that superclass's methods
        # Calling the previously built methods with supper() save you from having to rewrite them again in the subclass
        # The following variables:
        # self.training = True
        # slef._parameters = OrderedDict()
        # self._buffers = OrderedDict()
        # self._backward_hooks = OrderedDict()
        # self._forward_hooks = OrderedDict()
        # self._forward_pre_hooks = OrderedDict()
        # self._state_dict_hooks = OrderedDict()
        # self._modules = OrderedDict()
        #
        # super() does not return a method. It returns a proxy object. This object delgates calls to the correct class methods
        # without making an additional object to do so
        #
        #        lets you avoid referring to the base class explicitly, which can be nice. But the main advantage comes with multiple inheritance, where all sorts of fun stuff can happen. 
        #        See the standard docs on super if you haven't already.
        #
        #        Note that the syntax changed in Python 3.0: you can just say super().__init__() instead of super(ChildB, self).__init__() which IMO is quite a bit nicer. The standard docs also refer to a guide to using super() which is quite explanatory
        #       
        #       
        #       super(type[, object-or-type])
        #       
        #       
        #       
        #        Return a proxy object that delegates method calls to a parent or sibling class of type. This is useful for accessing inherited methods that have been overridden in a class. The search order is same as that used by getattr() except that the type itself is skipped.
        #
        #        The __mro__ attribute of the type lists the method resolution search order used by both getattr() and super(). The attribute is dynamic and can change whenever the inheritance hierarchy is updated.
        #
        #        If the second argument is omitted, the super object returned is unbound. If the second argument is an object, isinstance(obj, type) must be true. If the second argument is a type, issubclass(type2, type) must be true (this is useful for classmethods).
        #
        #        Note super() only works for new-style classes.
        #        There are two typical use cases for super. In a class hierarchy with single inheritance, super can be used to refer to parent classes without naming them explicitly, thus making the code more maintainable. This use closely parallels the use of super in other programming languages.
        #
        #        The second use case is to support cooperative multiple inheritance in a dynamic execution environment. This use case is unique to Python and is not found in statically compiled languages or languages that only support single inheritance. This makes it possible to implement diamond diagrams
        #        where multiple base classes implement the same method. Good design dictates that this method have the same calling signature in every case (because the order of calls is determined at runtime, because that order adapts to changes in the class hierarchy, and because that order can include 
        #        sibling classes that are unknown prior to runtime).
        #
        #        For both use cases, a typical superclass call looks like this:
        #
        #        class C(B):
        #            def method(self, arg):
        #                super(C, self).method(arg)
        #        Note that super() is implemented as part of the binding process for explicit dotted attribute lookups such as super().__getitem__(name). It does so by implementing its own __getattribute__() method for searching classes in a predictable order that supports cooperative multiple inheritance. 
        #        Accordingly, super() is undefined for implicit lookups using statements or operators such as super()[name].
        #
        #        Also note that super() is not limited to use inside methods. The two argument form specifies the arguments exactly and makes the appropriate references.
        #
        #        For practical suggestions on how to design cooperative classes using super(), see guide to using super().
        #       
        #       
        #       https://docs.python.org/2/library/functions.html#super 
        #
        # lets create out hiddenlayers 
        self.fcl1 = nn.Linear(in_Features,H1) # fully_connecte_layer #1
                                                # we are connecting in_features to H1 hidden layer
        self.fcl2 = nn.Linear(H1,H2)            # we are connecting hidden layer H1 to Hidden layer H2
        self.out = nn.Linear(H2,out_Features)    # We are connecting hidden layer H2 to the output layer o1
        #
        #
        #
        # lets initialize weights and biases
        #
        # initializing all weights to 0 can lead to slow convergence, as all the weights will be updated in 
        # the same direction. This can also cause the "Vanishing gradient issue"
        #nn.init.zeros_(self.fcl1.weight)
        #
        # initializing all weights to 1 can lead to slow convergence, as all the weights will be updated in 
        # the same direction. This can also cause the "Exploding gradient issue"
        #nn.init.ones_(self.fcl1.weight)
        ##
        #print("Standard Initialization of Dense Layer 1")
        #print(self.fcl1)
        #print("Standard Initialization of Weights of Dense Layer 1")
        #print(self.fcl1.weight)
        ##
        # It can help prevent the "vanishing gradient" issue, as the distribution has a finte range and the weights 
        # are distributed evenly accross that range. However this method can suffer from the "exploding gradient" problem
        # if that range is too large 
        print("Uniform Initialization of Weights of Dense Layer 1") 
        nn.init.uniform_(self.fcl1.weight)
        print(self.fcl1.weight)
        #
        # #
        #
        ## It can help prevent the "exploding gradient" issue, as the distribution has a finte range and the weights 
        ## are distributed evenly around the mean 
        #print("Normal Initialization of Weights of Dense Layer 1") 
        #nn.init.normal_(self.fcl1.weight,mean=0, std=1)
        #print(self.fcl1.weight)
        #
        # #
        #
        ## It can help prevent the "vanishing gradient" issue, as it scales the weights such that the variance of the
        ## variance of the outputs is the same as the variance of the inputs
        #print("Xavier Initialization of Weights of Dense Layer 1")
        #nn.init.xavier_uniform_(self.fcl1.weight)
        #print(self.fcl1.weight)
        ##
        # It can help prevent the "vanishing gradient" issue, as it scales the weights such that the variance of the
        # variance of the outputs is the same as the variance of the inputs, taking into account the nonlinearity 
        # of the activation function
        #print("Kaiming Initialization of Weights of Dense Layer 1")
        #nn.init.kaiming_uniform_(self.fcl1.weight,a=0, mode="fan_in",nonlinearity = "relu")
        #print(self.fcl1.weight)
        #
        # #
        ##
        #print("Costumized Initialization of Weights of Dense Layer 1")
        #def custom_weights_f(parameter,lowerBound=-0.5,upperBound=0.5):
        #    nn.init.uniform_(parameter.weight,lowerBound,upperBound)
        #self.fcl1.apply(custom_weights_f)
        #print(self.fcl1.weight)
        ##
        # #
        #
        ##
        #print("Standard Initialization of Biases of Dense Layer 1")
        #print(self.fcl1.bias)
        #print("Standard Initialization of Dense Layer 2")
        #print(self.fcl2)
        #print("Standard Initialization of Weights of Dense Layer 2")
        #print(self.fcl2.weight)
        #print("Standard Initialization of Biases of Dense Layer 2")
        #print(self.fcl2.bias)
        #print("Standard Initialization of Output Dense Layer")
        #print(self.out)
        #print("Standard Initialization of Weights of Output Dense Layer")
        #print(self.out.weight)
        #print("Standard Initialization of Biases of Output Dense Layer")
        #print(self.out.bias)
        ##
        #
        print("Uniform Initialization of Biases of Dense Layer 1") 
        nn.init.uniform_(self.fcl1.bias)
        print(self.fcl1.bias)
        print("Standard Initialization of Dense Layer 2")
        print(self.fcl2)
        print("Uniform Initialization of Biases of Dense Layer 2") 
        nn.init.uniform_(self.fcl2.weight)
        print(self.fcl2.weight)
        print("Uniform Initialization of Biases of Dense Layer 2") 
        nn.init.uniform_(self.fcl2.bias)
        print(self.fcl2.bias)
        print("Standard Initialization of Output Dense Layer")
        print(self.out)
        print("Uniform Initialization of Biases of Dense Layer out") 
        nn.init.uniform_(self.out.weight)
        print(self.out.weight)
        print("Uniform Initialization of Biases of Dense Layer out") 
        nn.init.uniform_(self.out.bias)
        print(self.out.bias)
        #
        #    
        #    
    # Here we define an scheme for the forward pass / forward propagation stage
    def forward(self,x):
        # lets define the activation functions being used for the first hidden layer H1
        x = F.relu(self.fcl1(x))       # F.relu : rectified linear unit
        x = F.relu(self.fcl2(x))        # we use the output x from fcl1 as input for fcl2
        x = self.out(x)                 # this way we get the output values from the output layer
        return x # this function returns the final value of x, that is, the prediction y_hat based on the original value of x (original input)




########################################################################
########################################################################
########################################################################
"""
Plotting Functions
"""
def basisIrisPlot(dfk,figName,labels):
    #fig, axes = plt.subplots(ncols=2,nrows=3, sharex=True, sharey=True)
    fig, axes = plt.subplots( nrows = 2, ncols = 2, figsize = (20,14) )
    fig.tight_layout()
    #
    colors = ['b','r','g']
    plots = [(0,1),(2,3),(0,2),(1,3)]
    #
    for ii, axk in enumerate(axes.flat):
        for jj in range(3):
            x = dfk.columns[plots[ii][0]]
            y = dfk.columns[plots[ii][1]]
            axk.scatter( dfk[ dfk['target'] == jj ][x], dfk[dfk['target'] == jj ][y], color = colors[jj] )
            axk.set(xlabel = x, ylabel = y)
    #
    #
    fig.legend(labels = labels, loc = 3, bbox_to_anchor = (1.0,0.85) )
    plt.savefig(figName)
    #plt.show()
    #
    #
########################################################################
########################################################################


########################################################################
########################################################################
def plotLoss101(epochs,losses,figName):
    plt.figure(figsize=(20,14))
    plt.plot(range(epochs),losses)
    plt.title('Evolution os Loss Function with the Epoch Number' , size = 15)
    plt.xlabel('Epoch Number' , size = 12)
    plt.ylabel('Loss Function Value' , size = 12)
    plt.savefig(figName)
    #plt.show()


########################################################################
########################################################################
""" 
The Haversine formula determines the distance between 2 points in a sphere
"""
def haversine_distance(df,lat1,long1,lat2,long2):
    r = 6371 # average radius of the earth in km
    #
    # lets convert the latitude and longitude coordinates to radians
    phi1_lat = np.radians(df[lat1])
    phi2_lat = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_gamma = np.radians(df[long2]-df[long1])
    #
    a = np.sin( delta_phi/2 )**2 + np.cos( phi1_lat )*np.cos( phi2_lat )*np.sin( delta_gamma/2 )**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers
    return d
    #
    #
########################################################################
########################################################################
########################################################################
"""
This function take in new paramters from the user, performs all pre-processing steps and passes new data from our 
trainned model
"""
def test_data(mdl): # pass in the name of the new model
    # INPUT NEW DATA
    plat  = float(input('What is the pickup latitude?  '))
    plong = float(input('What is the pickup longitude? '))
    dlat  = float(input('What is the dropoff latitude?  '))
    dlong = float(input('What is the dropoff longitude? '))
    psngr = int(input('How many passengers? '))
    dt    = input('What is the pickup date and time?\nFormat as YYYY-MM-DD HH:MM:SS     ')
    #
    # PREPROCESS THE DATA
    dfx_dict = {'pickup_latitude':plat,'pickup_longitude':plong,'dropoff_latitude':dlat,
         'dropoff_longitude':dlong,'passenger_count':psngr,'EDTdate':dt}
    dfx = pd.DataFrame(dfx_dict, index=[0])
    dfx['dist_km'] = haversine_distance(dfx,'pickup_latitude', 'pickup_longitude',
                                        'dropoff_latitude', 'dropoff_longitude')
    dfx['EDTdate'] = pd.to_datetime(dfx['EDTdate'])
    #
    #for category in categorical_columns:
    #    df[category] = df[category].astype('category')
    # We can skip the .astype(category) step since our fields are small,
    # and encode them right away
    dfx['Hour'] = dfx['EDTdate'].dt.hour
    dfx['AMorPM'] = np.where(dfx['Hour']<12,0,1) 
    dfx['Weekday'] = dfx['EDTdate'].dt.strftime("%a")
    dfx['Weekday'] = dfx['Weekday'].replace(['Fri','Mon','Sat','Sun','Thu','Tue','Wed'],
                                            [0,1,2,3,4,5,6]).astype('int64')
    # CREATE CAT AND CONT TENSORS
    cat_cols = ['Hour', 'AMorPM', 'Weekday']
    cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                 'dropoff_longitude', 'passenger_count', 'dist_km']
    #
    # Standard procedure
    #
    xcats = np.stack([dfx[col].values for col in cat_cols], 1)
    xcats = torch.tensor(xcats, dtype=torch.int64)
    xconts = np.stack([dfx[col].values for col in cont_cols], 1)
    xconts = torch.tensor(xconts, dtype=torch.float)
    
    # PASS NEW DATA THROUGH THE MODEL WITHOUT PERFORMING A BACKPROP
    with torch.no_grad():
        z = mdl(xcats, xconts)
    print(f'\nThe predicted fare amount is ${z.item():.2f}')

########################################################################
########################################################################
########################################################################
#


df = pd.read_csv('../PYTORCH_NOTEBOOKS/data/NYCTaxiFares.csv')
rows, cols = np.shape(df)
#debug#print(f'THe data has {rows} rows and {cols} columns')
print(df.head(20))
print(df.info())
print(df.describe())
#
#
#
#
# lets go ahead and do some feature engineering by creating a new column with the travelled distance in kilometers
# this will be more usefull than the actual coordinates
# there should be some correlation between the distance travelled and the cab fare
# -> another example: there should be some correlation between the wind speed and average / maximum tension 
# ->                    check if there is some correlation between the wind direction and average / maximum tension
# ->                    check if there is some correlation between the wind speed / direction gradients and average / maximum tension
""" 
The Haversine formula determines the distance between 2 points in a sphere
"""
df['distance_km'] = haversine_distance(df,'pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude')
"""
Alternative
df['distance_km'].apply(haversine_distance,df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'])
"""
#
#
# lets convert the time stamp to some useful metric
# lets obtain a date-time object
df['pickup_time_of_day'] = pd.to_datetime(df['pickup_datetime'])
# the time is in UTC, since Newyork uses the Eastern time, we need to specify the time zone
df['EasterTimeZone'] = df['pickup_time_of_day'] - pd.Timedelta(hours=4)
#
# lets suppose that the hour of the day has an influence in the taxi fare
df['Hour_of_day'] = df['EasterTimeZone'].dt.hour
# lets suppose that pm or a time has an influence in the taxi fare, lets check if the hour is greater than 12
df['AM_or_PM'] = np.where(df['Hour_of_day'] < 12,'am','pm') # if the condition is true, it will retunrn 'am', otherwise it will return 'pm 
# lets suppose that the week day has an influence in the taxi fare
df['WeekDay'] = df['EasterTimeZone'].dt.strftime("%a") # this return the abbreviated day of the week: Mon, Tues, ...m Sat, Sun
#df['WeekDay'] = df['EasterTimeZone'].dt.dayofweek() # this return the abbreviated day of the week:   0     1           5   6
print(df.head(20))
print(df.info())
print(df.describe())
#
#
#
# lets select our categorical columns
categorical_columns = ['Hour_of_day','AM_or_PM','WeekDay']
y_col               = ['fare_amount']  # this column contains the labels
#
continuous_columns  = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'distance_km']
# If you plan to use all of the columns in the data table, there's a shortcut to grab the remaining continuous columns
#cont_cols = [col for col in df.columns if col not in categorical_columns + y_col]
#
# lets encode the categorical features into numerics
# lets change the hour of day into a categorical feature
for category in categorical_columns:
    df[category] = df[category].astype('category')
# now these columns have a special data type called category
print(df.dtypes)
print(df['Hour_of_day'].head())
# 
print('lets report the categories in df[\'AM_or_PM\']')
print(df['AM_or_PM'].cat.categories)
print('lets report the one-hot encoding of these categories')
print(df['AM_or_PM'].cat.codes)
print('lets report the one-hot encoding in df[\'WeekDay\']')
print(df['WeekDay'].cat.codes)
print('lets report the one-hot encoding in df[\'WeekDay\'] in a numpy array')
print(df['WeekDay'].cat.codes.values)
#
# lets extract the values of the categorical columns
hour    = df['Hour_of_day'].cat.codes.values # -> Changes the column and assings to each dinstinc categorical value an id number : [0,1,2...,23] 
ampm    = df['AM_or_PM'].cat.codes.values # -> Changes the column and assings to each dinstinc categorical value an id number : [0,1]
weekday = df['WeekDay'].cat.codes.values # -> Changes the column and assings to each dinstinc categorical value an id number : [0,1,2...,7]
#
# lets stack these arrays into columns with axis = 1
categories = np.stack([hour,ampm,weekday],axis = 1)
# the above lines are equival to this one where the process is automized with a list comprehension
categories101 = np.stack( [df[col].cat.codes.values for col in categorical_columns] , 1) # 1 ->  axis = 1
"""
print(categories101[:5])
array([[ 4,  0,  1],
       [11,  0,  2],
       [ 7,  0,  2],
       [17,  1,  3],
       [22,  1,  0]], dtype=int8)
"""
#
print('Lets convert the categorical features into a tensor')
categories101_tensor = torch.tensor(categories101, dtype=torch.int64)
#categories101_tensor = torch.tensor(categories101, dtype=torch.int8)
print(categories101_tensor[:5])
"""
tensor([[ 4,  0,  1],
        [11,  0,  2],
        [ 7,  0,  2],
        [17,  1,  3],
        [22,  1,  0]])
"""
#
print('Lets stack the continuous features tool')
continuous101        = np.stack( [ df[col].values for col in continuous_columns ] , 1) # 1 ->  axis = 1
continuous101_tensor = torch.tensor(continuous101, dtype=torch.float)
"""
In order for the batch normalization to work properly, we need we have to store the continuous features and the target
feature as float (float32), not double (float64)

#continuous101_tensor = torch.tensor(continuous101, dtype=torch.float64)
#continuous101_tensor = torch.tensor(continuous101, dtype=torch.float16)
"""
print(continuous101_tensor)
print('lets conbert the column containing the target values or labels into a tensor too')
y_feature = torch.tensor( df[y_col].values, dtype=torch.float ).reshape(-1,1) # .reshape(-1,1) this ensures that we obtain a column vector instead of a row vector
#y_feature = torch.tensor( df[y_col].values, dtype=torch.float64 ).reshape(-1,1)
#debug#print(f'Input shapes: \ncategories {categories101_tensor.shape} \ncontinuous values {continuous101_tensor.shape} \n target values {y_feature.shape}')
_, n_categories101     = categories101_tensor.shape 
_, n_continuous101     = continuous101_tensor.shape 
_, n_targetFeatures101 = continuous101_tensor.shape 
#debug#print(f'Number of categorical features:     {n_categories101}')
#debug#print(f'Number of continuous features:      {n_continuous101}')
#debug#print(f'Number of target features:          {n_targetFeatures101}')
#
#
print('\n\n Now we need to define the embedding layer that comes with pytorch')
# it creates a lookup table that stores embeddings of a fixed dictionary and size 
# it is one-hot encoding that is applied to the categories
# the categorical data will be filtered through these embeddings in the forward section
#
print("lets store the sizes of our category columns")
category_sizes = [ len(df[col].cat.categories) for col in categorical_columns ] 
# for each categorical column, checks how many unique categories there are.
#
print(category_sizes)
# The rule of thumb for determining the embedding size is to divide the number of unique entries in each column by 2, but not to exceed 50.
embedding_sizes101 = [ ( size,min(50,(size+1)//2) ) for size in category_sizes ] # // ensures that there no floats after the division by 2
                                                                              # with this we are creating a tuple: ( size,min(50,(size+1)//2)
print(embedding_sizes101)
#
# 
# #
#
print('\n\n Now we will define the tabular model: \n - Embedding layer \n - Drop-out function \n - Normalization function \n - Setting up a sequential list of layers')
print('\nEnmbedding')
listOfEmbeddings = [nn.Embedding(ni,nf) for ni,nf in embedding_sizes101 ] # we are defining embedding for the tuples values of ( size,min(50,(size+1)//2)
print( listOfEmbeddings ) # we are defining embedding for the tuples values of ( size,min(50,(size+1)//2)
# ni is the number of categories
# nf is the number of dimensions
#
#
# #
#
print("Generating Self Embeddings")
selfEmbeddings = nn.ModuleList( listOfEmbeddings  )
print(selfEmbeddings)
# The previous steps are ussually encapsulated inside the constructor of the Tabular model class
#
#
# #
#
print('lets create a list of tuple values with these embeddings')
listOfEmbeddingsTuples = list( enumerate( selfEmbeddings ) )
print(listOfEmbeddingsTuples)
#
#
# #
#
# Inside the forward method we need to pass our sources of categorical data
EmbeddingsSourcesList = []
for ii, ee in enumerate( selfEmbeddings ):
    EmbeddingsSourcesList.append( ee(categories101_tensor[:,ii]) )
print("The embeddings have assinged values to the one-hot encodings of the categorical features")
print(EmbeddingsSourcesList)    
# 
# 
print('In the forward loop lets create a categorical input z \n by concatenating the embedding sections (12,1,4) into one (17)')
z = torch.cat(EmbeddingsSourcesList, 1)
print(z)
# 
print('The next step is to pass input z though a drop out layer that randomly drops / sets to 0 some of z elemts')
print('This helps to avoind overffiting')
dropOutProbability = 0.4 # 0.6 #0.5
selfEmbeddingDropLayer = nn.Dropout(dropOutProbability)
z = selfEmbeddingDropLayer(z) # passing z vector (torch tensor, actually) trough the drop out layer
# this is how the categorical features are being passed down through the layers
# 
# 
# 
class TabularRegressionModel101(nn.Module):
    # 
    # Constructor / instantiation method
    #def __init__(self, embedding_sizes, n_categories, n_continuous, n_outputSize, dropOutProbability = 0.5 ):
    def __init__(self, embedding_sizes, n_continuous, n_outputSize, layers, dropOutProbability = 0.5 ):
        # embedding_sizes: dimensions of the embding layers for filtering categorical data
        #                       list of tuples: each categorical variable size is paired with an embedding size
        #                       
        # n_categories: number of columns of categorical feature variables in the data
        # n_continuous : number of columns of continuous feature variables in the data
        # n_outputSize : number of columns of output / target features matrix in the data
        # layers: layer sizes: definition of the number layers and neurons per layer.   layerVec = [n1, n2, n3, ...] -> len(layerVec) number of hidden layers architecture
        #           n1: number of nurons in hidden layer #1,
        #           n2: number of nurons in hidden layer #2,
        #           n3: number of nurons in hidden layer #3,
        #           ...
        # dropOutProbability: dropout probability for each layer (for simplicity we'll use the same value throughout)
        # lets instatiate the mother class
        super().__init__() 
        #
        """
        Defintion of the type of layers
            - self.Embeddings                       -> filtering layers for categorical features
            - self.embeddingDropLayer               -> layer for dropping out weights and biases of some neurons
            - self.batchNormalizedContinuousData    -> layers normalizing initial continuous data only
            - layerList
        """
        #print("\nGenerating Self Embedding Layers for Filtering Categorical Feature Values") # we are defining embedding for the tuples values of ( size,min(50,(size+1)//2)
        #listOfEmbeddings = [nn.Embedding(ni,nf) for ni,nf in embedding_sizes ] # ni is the number of categories, nf is the number of dimensions
        """print( listOfEmbeddings ) # we are defining embedding for the tuples values of ( size,min(50,(size+1)//2)
        #[Embedding(24, 12), Embedding(2, 1), Embedding(7, 4)]
        """
        #
        #self.Embeddings = nn.ModuleList( listOfEmbeddings  )
        #
        # all toguether:
        self.Embeddings = nn.ModuleList( [nn.Embedding(ni,nf) for ni,nf in embedding_sizes ]  )
        """print(selfEmbeddings)
        #ModuleList(
        #  (0): Embedding(24, 12)
        #  (1): Embedding(2, 1)
        #  (2): Embedding(7, 4)
        #)"""
        #
        #print("\nGenerating Drop-Out Layer")
        self.embeddingDropLayer = nn.Dropout(dropOutProbability)
        #
        #print("\nNormalizing Data to Similar Order of Magnitude Range")
        # by normalizing continuous data the internal covariate shift is reduced and this helps to accelerate training: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,Sergey Ioffe,Google Inc., sioffe@google.com,Christian Szegedy,Google Inc., szegedy@google.com.
        self.batchNormalizedContinuousData = nn.BatchNorm1d(n_continuous) #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # -> check other types of normalizations
        # 
        # 
        # 
        layerList           = [] #variable to store layers, 
        """
        each element repsents a disticn layer type emplying the number of neurons defined in the ii-th element 
        of variable "layers" 
        """
        numberOfEmbeddings  = sum( [nf for ni, nf in embedding_sizes] ) # 12 + 1 + 4 ... sum of the number of dimensions (subcategories) of each categorical column
        numberOfInputs      = numberOfEmbeddings + n_continuous
        # 
        #print("\nHidden layers definition")
        for ii in layers:
            # lets define linear layers (layer fully connected) for now
            layerList.append(nn.Linear(numberOfInputs,ii)) #Applies a linear transformation to the incoming data
            layerList.append(nn.ReLU(inplace=True)) # try with leaky relu since avoids vanishing gradient issue
            layerList.append(nn.BatchNorm1d(ii)) # layer for normalizing inputs of hidden layers, taking into account continuous and embedded categorical features 
            layerList.append(nn.Dropout(dropOutProbability)) # layer for randomly making zero some wights and bias of inner layers
            numberOfInputs = ii # the number of outpust of a previous layer is equal to the number of imputs of the next layers
        # 
        #print("\nFinal layer definition")
        layerList.append(nn.Linear(layers[-1],n_outputSize))
        # layer[-1] -> last row of the matrix defining hidden layer architecture
        # n_outputSize -> for regression problems is always equal to 1, since we want to estimate magnitude values
        # 
        # Creating a sequentieal container of layers in which the output of layer k-1 is chained sequentilly to layer k 
        # as it was defined in the imput list *layerList
        self.layers = nn.Sequential(*layerList)
        """
        A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        Alternatively, an OrderedDict of modules can be passed in. The forward() method of Sequential accepts 
        any input and forwards it to the first module it contains. It then chains outputs to inputs sequentially 
        for each subsequent module, finally returning the output of the last module. 

        # lets define the activation functions being used for the first hidden layer H1
        x = F.relu(self.fcl1(x))       # F.relu : rectified linear unit
        x = F.relu(self.fcl2(x))        # we use the output x from fcl1 as input for fcl2
        x = self.out(x)                 # this way we get the output values from the output layer
        return x # this function returns the final value of x, that is, the prediction y_hat based on the original value of x (original input)

        """
        # #
    #
    # forward method
    def forward(self, x_categories_tensor101, x_continuous_tensor101):
        """
        Defintion of the architecture of the forward pass 
        """
        # x_categorical     = categorical feature tensor
        # x_continuous      = continuous feature tensor
        #
        # embedding layer stage
        EmbeddingsSourcesList = []
        for ii, ee in enumerate(self.Embeddings): # #  ii = (0) : ee = Embedding(24, 12)
            EmbeddingsSourcesList.append( ee(x_categories_tensor101[:,ii]) ) # " : " sweeps all the rows of the x_categories_tensor101 (the same as in the data frame) and " ii " inidicates the id of the category
        x_features = torch.cat(EmbeddingsSourcesList, 1) # passing embedded categorical feature matrix to the general feature matrix
        x_features = self.embeddingDropLayer(x_features) # passing through the drop-out stage 
        # normalizing continuous feature matrix
        x_continuous_tensor101 = self.batchNormalizedContinuousData(x_continuous_tensor101)
        # merging categorical data with continuous data
        x_features = torch.cat([x_features,x_continuous_tensor101],1)
        # passing through all the layers for make prediction "y_hat = a"
        return self.layers(x_features)
    #
    # 
    # 
    # #
# 
# 
# 
# 
# #
#lets use random numbers to initialize the weights and biases
torch.manual_seed(33) # theis way evvrytime we weill get the same random numbers with sedd 32
#
# lets intantiate the tabular model
regressionModel101 = TabularRegressionModel101(embedding_sizes101, n_continuous101, 1, [100,200], dropOutProbability = 0.4 )
print(regressionModel101)
# lets visualize the data
figName             = "aaa_lossEvountion.png"
figNameLoss         = "Evolution of the loss function vs number of epochs"
labels              = ['xxx','xxx','xxx']
modelName103        = "TaxiFareRegrModel.pt"
#
#lets create a folder for saving models
path1 = './models/PyTorch/'
os.makedirs(path1, exist_ok=True)
modelName101        =path1+"taxiFareRegression101.pt" # .pt files are binary files with PyTorch data
modelName102        =path1+"taxiFareRegression101_pickleFile.pt" # .pt files are binary files with PyTorch data

#
# lets define a loss function:
MSE_criterion  = nn.MSELoss() # mean square error
def RMSE_criterion(y_hat,y_feature):
    return torch.sqrt(MSE_criterion(y_hat,y_feature)) # root mean square error
# -> check:    = apply(torch.sqrt(),MSE_criterion)
#
# lets define an algorithm for the optimizer 
optimizer = torch.optim.Adam(regressionModel101.parameters(), lr = 0.001)
#
# lets split the data:
batch_size = 60000
test_size = int(batch_size * .2)
# 
X_cat_train = categories101_tensor[:batch_size-test_size]
X_cat_test  = categories101_tensor[batch_size-test_size:batch_size]
X_con_train = continuous101_tensor[:batch_size-test_size]
X_con_test  = continuous101_tensor[batch_size-test_size:batch_size]
y_train   = y_feature[:batch_size-test_size]
y_test    = y_feature[batch_size-test_size:batch_size]
#
#
# lets train the model
start_time = time.time()
#
# as a general rule, start with only a few epochs (1 run through all the training data), specially if you have a large data set and plot the loss function and then decide if more epochs are needed
epochs = 300
#losses = [] # with this we track the evolution of the loss function
losses = np.zeros(epochs)
#print(len(X_train_torch),len(y_train_torch))
for ii in range(epochs):
    ii+=1
    #y_hat_train = regressionModel101(X_cat_train,X_con_train)
    y_hat_train = regressionModel101.forward(X_cat_train,X_con_train)
    #loss = MSE_criterion(y_hat_train, y_train)) # MSE
    #loss = torch.sqrt(criterion(y_hat_train, y_train)) # RMSE
    loss = RMSE_criterion(y_hat_train, y_train) # RMSE
    losses[ii-1] = loss
    #losses.append(loss)
    #
    # a neat trick to save screen space:
    if (ii-1)%25 == 1:
    #if(ii%10==0):
        #print(f'epoch # {ii} \t loss value: {loss:10.3f} weight: {model101.Linear.weight.item():10.8f} bias: {model101.linear.bias.item():10.8f}')
        #print(f'epoch # {ii+1} \t loss value: {loss:10.8f}')
        #debug#print(f'epoch: {ii:3}  loss: {loss.item():10.8f}')
        print("Epoch #",ii,", loss:",loss)
        
    #
    # backpropagation stage
    # lets find where the gradient is zero, since we are looking for minimums 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #
#debug#print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
#debug#print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
#
#
# lets plot the evolution of loss function:
plotLoss101(epochs,losses,figNameLoss)
print("min loss during training: ",min(losses),"\tlast loss: ",losses[-1])
# Target value: 3.4246
# Actual value: 3.6117
#
#
# TO EVALUATE THE ENTIRE TEST SET
with torch.no_grad():
    #y_hat_test = regressionModel101(X_cat_test,X_con_test)
    y_hat_test = regressionModel101.forward(X_cat_test,X_con_test)
    #loss = MSE_criterion(y_hat_test, y_test)) # MSE
    #loss = torch.sqrt(criterion(y_hat_test, y_test)) # RMSE
    loss = RMSE_criterion(y_hat_test, y_test) # RMSE
    #
#debug#print(f'RMSE: {loss:.8f}')
print("RMSE: ",loss)
# Target value: 3.3459
# Actual value: 3.5021
#
# 
# 
# 
# #
#
# Check for the largets and smallest diffrencen between the predicted and tur values
print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
for i in range(50):
    diff = np.abs(y_hat_test[i].item()-y_test[i].item())
    print(f'{i+1:2}. {y_hat_test[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')
# 
print("Maximum difference: ",max(abs(y_hat_test-y_test)))
# 
# 
# 
# #
#
# Make sure to save the model only after the training has happened!
if len(losses) == epochs:
    torch.save(regressionModel101.state_dict(), regressionModel101)
else:
    print('Model has not been trained. Consider loading a trained model instead.')

"""
Before we can load the saved model, we need to instantiate the current tabular model with the 
the current parameters:
- Embedding sizes
- Number of continuous columns
- Output size
- layer sizes
- dropout value
"""
#emb_szs = [(24, 12), (2, 1), (7, 4)]
#model2  = regressionModel101(emb_szs, 6, 1, [200,100], p=0.4)
## now lets load the model
#model2.load_state_dict(torch.load(modelName101))
#model2.eval() # be sure to run this step!