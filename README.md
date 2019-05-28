# Generating-Code-Snippet-from-Natural-Language
dependency :  

tensorflow  
python3  
astunparse  
conda install -c conda-forge astunparse  
conda install -c conda-forge/label/gcc7 astunparse  
conda install -c conda-forge/label/cf201901 astunparse  


generate data  
  sh init.sh  
train  
  python3 train.py -c(default)|-h -p(default)|-np -s(default)|-ns  
predict  
  python3 predict.py -c|-h -p|-np -s|-ns  
evaluate  
  python3 evaluate.py -c|-h -p|-np -s|-ns  

-c : using conala dataset  
-h : using hs dataset  
-p : using pre train  
-np : not using pre train  
-s : using semantic logic order  
-ns : not using semantic logic order  