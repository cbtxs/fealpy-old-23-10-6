import numpy as np

class Array2dWithDiffCol(object):
    def __init__(self, array, location):
        self.array = array
        self.loc = location

    def __getitem__(self, index):
        # 对切片进行处理，返回特定列的数据
        if isinstance(index, tuple):
            row_slice, col_slice = index
            if col_slice == 1:
                return self.get_column(row_slice)
        return self.array[index]

    def __setitem__(self, index):
        # 对切片进行处理，返回特定列的数据
        if isinstance(index, tuple):
            row_slice, col_slice = index
            if col_slice == 1:
                return self.get_column(row_slice)
        return self.array[index]

    def get_column(self, column_index):
        # 获取特定列的数据
        start = self.loc[column_index]
        end = self.loc[column_index + 1] if column_index + 1 < len(self.loc) else len(self.array)
        return self.array[start:end]

# 创建一个示例数组
array_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 创建 Array2dWithDiffCol 对象
locations = [0, 3, 6]
array_2d = Array2dWithDiffCol(array_data, locations)

# 使用自定义的切片操作
column_data = array_2d[:, 1]

print("Column Data:")
print(column_data)
        
