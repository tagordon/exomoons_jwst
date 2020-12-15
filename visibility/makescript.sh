awk '{print "jwst_gtvt " $2 " " $3 " --name \"" $1 "\" --save_plot " $1 ".jpg --save_table " $1".dat"}' params.txt > calcvis.sh
