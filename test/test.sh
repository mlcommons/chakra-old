# Need to be the same with updater 
NUM_NPU=4
NUM_DIMS=2

mkdir tmp 
python3 -m utils.et_generator.et_generator \
    --num_npus $NUM_NPU \
    --num_dims $NUM_DIMS
mv *.et tmp/

# Test for generator
for entry in test/standard_output/et/*
do
  file=$(basename -- "$entry")
  echo Comparing ET: $file
  STATUS="$(cmp --silent $entry tmp/$file; echo $?)"  # "$?" gives exit status for each comparison

  if [[ $STATUS -ne 0 ]]; then
    echo Fail 
    exit 1 # Raise an error 
  fi
  echo Pass 
done

# Test for visualizer 
for entry in test/standard_output/et/*
do
  file=$(basename -- "$entry")
  python3 -m et_visualizer.et_visualizer\
      --input_filename $entry\
      --output_filename ./tmp/$file.pdf

  echo Check if PDF exists: $file.pdf

  if [ -f "tmp/$file.pdf" ]; then
    echo Pass 
  else
    exit 1 # Raise an error 
  fi
done

rm -rf tmp
