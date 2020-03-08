

fib_data_filenames = ['{}.txt'.format(i) for i in range(9)]



all_num = 0
tmp = 0
for filename in fib_data_filenames:
    with open(filename, 'r') as f:
        for line in f:
            nums = line.split(',')
            nums = list(map(int, nums))
            if tmp:
                nums[0] += tmp
                tmp = 0
            if len(nums) % 2:
                nums, tmp = nums[:-1], nums[-1]
            with open('data3k_2.txt', 'a') as f:
                for num in nums:
                    f.write('{}\n'.format(num))
                    all_num += num

#print(all_num, tmp) # 43046721 0
#print(sum(map(int, nums)), 9**8) # 43046721 43046721
