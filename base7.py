def tobase7(n):
    ans = ""
    while (n>0):
        remainder = n % 7
        n = n // 7
        ans = str(remainder) + ans
    return ans

print(tobase7(10))
print(tobase7(100))
print(tobase7(1000))