grep -v "Trial,Fold,AUC" cat_pca_fr_baby.csv > temp && mv temp cat_pca_fr_baby.csv
grep -v "Trial,Fold,AUC" cat_pca_fn_baby.csv > temp && mv temp cat_pca_fn_baby.csv
grep -v "Trial,Fold,AUC" cat_pca_fn_mom.csv > temp && mv temp cat_pca_fn_mom.csv
grep -v "Trial,Fold,AUC" cat_pca_fn_all.csv > temp && mv temp cat_pca_fn_all.csv
grep -v "Trial,Fold,AUC" cat_fn_all.csv > temp && mv temp cat_fn_all.csv
grep -v "Trial,Fold,AUC" cat_fn_mom.csv > temp && mv temp cat_fn_mom.csv
grep -v "Trial,Fold,AUC" cat_fn_baby.csv > temp && mv temp cat_fn_baby.csv
grep -v "Trial,Fold,AUC" cat_fr_baby.csv > temp && mv temp cat_fr_baby.csv

grep -v "Trial,Fold,AUC" mlp_pca_fr_baby.csv > temp && mv temp mlp_pca_fr_baby.csv
grep -v "Trial,Fold,AUC" mlp_pca_fn_baby.csv > temp && mv temp mlp_pca_fn_baby.csv
grep -v "Trial,Fold,AUC" mlp_pca_fn_mom.csv > temp && mv temp mlp_pca_fn_mom.csv
grep -v "Trial,Fold,AUC" mlp_pca_fn_all.csv > temp && mv temp mlp_pca_fn_all.csv
grep -v "Trial,Fold,AUC" mlp_fn_all.csv > temp && mv temp mlp_fn_all.csv
grep -v "Trial,Fold,AUC" mlp_fn_mom.csv > temp && mv temp mlp_fn_mom.csv
grep -v "Trial,Fold,AUC" mlp_fn_baby.csv > temp && mv temp mlp_fn_baby.csv
grep -v "Trial,Fold,AUC" mlp_fr_baby.csv > temp && mv temp mlp_fr_baby.csv

grep -v "Trial,Fold,AUC" mlp_fn_all_1_10.csv > temp && mv temp mlp_fn_all_1_10.csv
grep -v "Trial,Fold,AUC" mlp_fn_mom_1_10.csv > temp && mv temp mlp_fn_mom_1_10.csv
grep -v "Trial,Fold,AUC" mlp_fn_baby_1_10.csv > temp && mv temp mlp_fn_baby_1_10.csv

grep -v "Trial,Fold,AUC" mlp_fn_all_1_30.csv > temp && mv temp mlp_fn_all_1_30.csv
grep -v "Trial,Fold,AUC" mlp_fn_mom_1_30.csv > temp && mv temp mlp_fn_mom_1_30.csv
grep -v "Trial,Fold,AUC" mlp_fn_baby_1_30.csv > temp && mv temp mlp_fn_baby_1_30.csv

grep -v "Trial,Fold,AUC" mlp_fn_all_3_10.csv > temp && mv temp mlp_fn_all_3_10.csv
grep -v "Trial,Fold,AUC" mlp_fn_mom_3_10.csv > temp && mv temp mlp_fn_mom_3_10.csv
grep -v "Trial,Fold,AUC" mlp_fn_baby_3_10.csv > temp && mv temp mlp_fn_baby_3_10.csv

grep -v "Trial,Fold,AUC" mlp_fn_all_3_30.csv > temp && mv temp mlp_fn_all_3_30.csv
grep -v "Trial,Fold,AUC" mlp_fn_mom_3_30.csv > temp && mv temp mlp_fn_mom_3_30.csv
grep -v "Trial,Fold,AUC" mlp_fn_baby_3_30.csv > temp && mv temp mlp_fn_baby_3_30.csv