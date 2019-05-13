(start-idyom)

(idyom-db:delete-dataset 12)
(idyom-db:import-data :krn "FOLDER" "Temporary dataset for evluation" 12) 

(idyom-db:export-data (idyom-db:get-dataset 12) :mid "lisp/midis/")

(idyom:idyom 12 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./lisp/" :overwrite t)
(quit)
