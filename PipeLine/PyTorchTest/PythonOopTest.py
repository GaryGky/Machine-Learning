# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:31:36 2019

@author: dell
"""

class Pet:
    def __init__(self,name,age,sex):
        self.name = name
        self.age = age
        self.sex = sex
    
    def printName(self):
        print("This is %s" % self.name)
    
    def selfIntroduction(self):
        print("This is %s. I am %d years old. And I am a %s" % 
              (self.name, self.age, self.sex))
    
class Cat(Pet):
    def selfIntroduction(self):
        print("I am a Cat!")
        print("This is %s. I am %d years old. And I am a %s" % 
              (self.name, self.age, self.sex))
class Dog(Pet):
    def selfIntroduction(self):
        print("I am a Dog!")
        print("This is %s. I am %d years old. And I am a %s" % 
              (self.name, self.age, self.sex))

cat = Cat("Tom",10,"Boy")
dog = Dog("Sponge",5,"Girl")

cat.selfIntroduction()
dog.selfIntroduction()