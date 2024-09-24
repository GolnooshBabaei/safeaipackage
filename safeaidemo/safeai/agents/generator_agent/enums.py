from safeai.enums import SafeAIStrEnum

class Emotion(SafeAIStrEnum):
    """Emotion Labels for Sentiment Analysis"""
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"

class CategoryType(SafeAIStrEnum):
    """_summary_"""
    CONTINENT = "continent"
    COUNTRY = "country"
    STATE = "state"
    TOWN = "town"
    GENDER = "gender"
    RANDOM = "random"
    

class ColumnDataType(SafeAIStrEnum):
    """Data Types for Input Data Configuration"""
    TEXT = "text"
    INTEGER = "number"
    DECIMAL = "decimal"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class TextType(SafeAIStrEnum):
    """_summary_"""
    URL = "url"
    TWEET = "Tweet"
    EMAIL = "Email"
    NEWS_ARTICLE = "News Article"
    BLOG_ARTICLE = "Blog Article"
    REVIEW = "Review"
    INSTAGRAM_COMMENT = "Instagram Comment"
    FACEBOOK_COMMENT = "Facebook Comment"
    PERSONAL_EMAIL_ADDRESS = "personal_email_address"
    WORK_EMAIL_ADDRESS = "work email address"
    PHONE_NUMBER = "phone number"
    CREDIT_CARD_NUMBER = "credit card number"
    FIRST_NAME = "first name"
    LAST_NAME = "last name"
    PASSWORD = "password"
    USERNAME = "username"

class TimeFrequency(SafeAIStrEnum):
    """Time Frequency for Time Series Data"""
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"
    YEARLY = "Yearly"
    HOURLY = "Hourly"
    MINUTELY = "Minutely"
    SECONDLY = "Secondly"
    MILLISECONDLY = "Millisecondly"