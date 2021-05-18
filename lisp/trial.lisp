(idyom-db:delete-dataset 12)

(idyom-db:import-data :krn "lisp/Data/chorales_krn/" "Temporary dataset for evluation" 12) 

(idyom:idyom 13 '(cpitch onset) '(cpitch onset) :k 1 :pretraining-ids '(12) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)

