(start-idyom)

(idyom-db:delete-dataset FOLDER2)
(idyom-db:import-data :krn "FOLDER" "Temporary dataset for evluation" FOLDER2) 

(idyom-db:export-data (idyom-db:get-dataset FOLDER2) :mid "lisp/midis/")

(idyom:idyom FOLDER2 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./lisp/" :overwrite t)
(quit)


; (idyom:idyom FOLDER3 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)


; (idyom:idyom FOLDER3 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./lisp/" :overwrite t)


; (idyom:idyom FOLDER3 '(cpitch onset) '(cpitch onset) :k FOLDER :pretraining-ids '(FOLDER) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)


; (idyom:idyom FOLDER2 '(cpitch onset) '(cpitch onset) :pretraining-ids '(FOLDER2) :models :both :detail 3 :output-path "./lisp/" :overwrite t)

; (idyom:idyom FOLDER3 '(cpitch onset) '(cpitch onset) :k FOLDER :pretraining-ids '(FOLDER2) :models :both :detail 3 :output-path "./stimuli/giovanni/surprises/" :overwrite t)

; (idyom-db:import-data :krn "FOLDER" "Temporary dataset for evluation" FOLDER2) 



(idyom:idyom 3 '(cpitch onset) '(cpitch onset) :models :both :detail 3 :output-path "./" :overwrite t)