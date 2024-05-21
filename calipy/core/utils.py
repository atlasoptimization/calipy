#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides support functionality related to logging, data organization,
preprocessing and other functions not directly related to calipy core domains.

The classes are
    CalipyRegistry: Dictionary type class that is used for tracking identity
        and uniqueness of objects created during a run and outputs warnings.

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    CalipyRegistry class
"""


class CalipyRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, key, value):
        if key in self.registry:
            print(f"Warning: An item with the key '{key}' already exists.")
        self.registry[key] = value
        print(f"Item with key '{key}' has been registered.")

    def get(self, key):
        return self.registry.get(key, None)

    def remove(self, key):
        if key in self.registry:
            del self.registry[key]
            print(f"Item with key '{key}' has been removed.")
        else:
            print(f"Item with key '{key}' not found in registry.")

    def clear(self):
        self.registry.clear()
        print("Registry has been cleared.")

    def list_items(self):
        for key, value in self.registry.items():
            print(f"{key}: {value}")