# Need to be the same with tester
NUM_NPU=4
NUM_DIMS=2

rm test/standard_output/et/*
rm test/standard_output/pdfs/*

python3 -m utils.et_generator.et_generator \
    --num_npus $NUM_NPU \
    --num_dims $NUM_DIMS 

mv *.et test/standard_output/et/

for entry in test/standard_output/et/*
do
  file=$(basename -- "$entry")
  python3 -m et_visualizer.et_visualizer\
      --input_filename $entry\
      --output_filename test/standard_output/pdfs/$file.pdf
done
