import sys
import os
import re
# TODO: We should compile a list of questions for our meeting with Cartledge about what he expects from this program. \
#  - Does he just want command line prompts or a basic GUI?                                                           \
#  - What does he expect his input to look like?                                                                      \
#  - What does he expect the output to be?                                                                            \
#  -


class Planet:
    def __init__(self, name):
        self.name = name
        self.data_array = []

    # Defines the string representation of the Planet class as the planet name
    def __repr__(self):
        return str(self.name)

    # Adds a Dataset object to the planet data array
    def add_dataset(self, filename):
        data = Dataset()
        data.parse_data(filename)
        self.data_array.append(data)


class Dataset:
    def __init__(self):
        self.data_quality = None
        self.epoch_array = []
        self.depth_array = []
        self.error_array = []

    # Parses a data file and populates the Dataset class attributes
    def parse_data(self, filename):
        # TODO: Figure out how to make pycharm ignore specific style issues. It feels the escaping of these periods \
        #  is a style issue and I want it to ignore this specific case but not any other styling issues.
        # Gets the data quality from the filename
        self.data_quality = re.findall("\d\.\d*[^\.]", filename)[0][0]
        for line in open(filename, 'r'):
            # If the line doesn't start with a digit discard it
            if not line[0].isdigit():
                continue
            # TODO: Currently this is getting around different data formats with an ugly try-except and adding any \
            #  extra info into the error field. This is gross and I hate it
            try:
                # Unpacks the array returned from the split function into temp variables
                # The * in *error allows for a variable number of values to be saved in error as an array
                epoch, depth, *error = line.split()

            # if < 3 values in line set error to None
            except ValueError:
                epoch, depth = line.split()
                error = None
            self.epoch_array.append(epoch)
            self.depth_array.append(depth)
            self.error_array.append(error)


# TODO: Begin trying different SciPy things on this data and seeing what comes out
if __name__ == '__main__':
    planet_array = []
    # Iterate through each file in label_data
    for subdir, dirs, files in os.walk('label_data'):
        try:
            # The main directory (label_data) has no subdir and will throw an exception when trying to index after
            # the split. This is fine since we want to skip it anyways
            planet_name = subdir.split('/')[1]
            planet = Planet(planet_name)
            for file in files:
                file.encode("utf-8")
                # add each file within a planets directory as a dataset
                planet.add_dataset(os.path.join(subdir, file))
                # print(os.path.join(subdir, file))
            planet_array.append(planet)
        except IndexError:
            continue

    # Test prints
    print("Planet Name: ", planet_array[0])
    print("Epoch Time: ", planet_array[0].data_array[0].epoch_array[0])
    print("Depth of Transit: ", planet_array[0].data_array[0].depth_array[0])
    print("Error/Other-Data: ", planet_array[0].data_array[0].error_array[0])
    print("Data Quality: ", planet_array[0].data_array[0].data_quality)

