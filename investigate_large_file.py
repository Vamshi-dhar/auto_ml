import bz2
import cPickle as pickle
import sys

# Borrowed directly from https://github.com/bosswissam/pysize/blob/master/pysize.py
import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))
    return size


with open("auto_ml_saved_pipeline.pkl", "rb") as read_file:
    trained_ml_pipeline = pickle.load(read_file)

# Recursive function to find where the large items are
def print_size_info(item, name=None):

    if get_size(item) > 10000:
        if name is not None:
            print('the name of the item')
            print(name)
        print('the item itself')
        print(item)
        print('the size of the above item')
        print(get_size(item))

        if isinstance(item, dict):
            for k, v in item.items():
                print_size_info(v, k)
        elif hasattr(item, '__dict__'):
            print_size_info(item.__dict__)
        elif hasattr(item, '__iter__') and not isinstance(item, (str, bytes, bytearray)):
            for i in item:
                print_size_info(i)


print('trained_ml_pipeline.named_steps')
print(trained_ml_pipeline.named_steps)
for step in trained_ml_pipeline.named_steps:
    print('new step!')
    print(step)
    print_size_info(trained_ml_pipeline.named_steps[step])
    # print('here is the item itself')
    # print(trained_ml_pipeline.named_steps[step])
    # print('here is the size of the item')
    # print(get_size(trained_ml_pipeline.named_steps[step]))

