EMBEDDING_LENGTH = 27
LETTER_MAP = {' ': 0}
ENCODING_MAP = [' ']
for i in range(26):
    LETTER_MAP[chr(ord('a') + i)] = i + 1
    ENCODING_MAP.append(chr(ord('a') + i))
LETTER_LIST = list(LETTER_MAP.keys())
'''
LETTER_MAP
{
    ' ': 0,
    'a': 1,
    'b': 2,
    ...
    'z': 26
}
LETTER_LIST
[' ', 'a', 'b', ..., 'z']
'''