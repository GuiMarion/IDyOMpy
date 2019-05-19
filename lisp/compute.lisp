(start-idyom)

(idyom-db:delete-dataset 12)
(idyom-db:import-data :krn "FOLDER" "Temporary dataset for evluation" 12) 

(idyom-db:export-data (idyom-db:get-dataset 12) :mid "lisp/midis/")

(idyom:idyom 12 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./lisp/" :overwrite t)
(quit)


; (idyom:idyom 13 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)


; (idyom:idyom 13 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./lisp/" :overwrite t)


; (idyom:idyom 13 '(cpitch onset) '(cpitch onset) :k 1 :pretraining-ids '(1) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)


; (idyom:idyom 12 '(cpitch onset) '(cpitch onset) :pretraining-ids '(12) :models :both :detail 3 :output-path "./lisp/" :overwrite t)

; (idyom:idyom 13 '(cpitch onset) '(cpitch onset) :k 1 :pretraining-ids '(12) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)

; (idyom-db:import-data :krn "FOLDER" "Temporary dataset for evluation" 12) 


