import src
print('src module:', src)
print('src path attr:', getattr(src, '__file__', None))
print('src path __path__:', getattr(src, '__path__', None))
