from enum import Enum

class ActionType(str, Enum):
    IGNORE = "IGNORE"
    WARN = "WARN"
    TIMEOUT_60S = "TIMEOUT_60S"
    TIMEOUT_600S = "TIMEOUT_600S"
    BAN = "BAN"