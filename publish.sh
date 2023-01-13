rm -rf dist
rm -rf build
rm -rf webuiapi.egg-info
python3 setup.py bdist_wheel
twine upload dist/*
