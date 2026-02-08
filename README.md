# Complex Granger Analisys Library
Library of causal analisys alorthims which was created as main subject of BSc thesis and further developted as part of MA thesis.  It implements various implementations of Granger Analisys algorithms with modifications.  It is also beening developed in a way that is supposed to allow easy customization and development

# Installing
It is recomended to clone repository or to download repository and place complex_granger_analysis folder in your project directory. You can use bellow code to clone the repository:

`git clone https://github.com/Smoldersob/complex_granger_analysis.git`



# Using
After importing/cloning you can access add library by importing it using import:

`import complex_granger_analysis as cga`

Or its parts using from:

`from complex_granger_analysis.granger_tests import tensorflow_granger`

Since torch and tensorflow are not recomended to be used in one enviroment, parts that use tensorflow or torch will be automatically included or excluded dependend on available library.

## Example of using with tensorflow

<pre>
  import complex_granger_analysis as CGA
  import pandas as pd
  
  data=pd.read_csv("example/PID_no_fault.csv",sep=";",index_col='Unnamed: 0')
  
  GCANN=CGA.TFNeuralSparseConstaraintedMVGC()
  GCANN.fit(data_df,causes=['u','f1','f2'],effects=['x1','x2','x3','x4'])
  causal_matrix=GCANN.results.result()
  causal_matrix.to_csv("PID_causal_relations")
  
</pre>
