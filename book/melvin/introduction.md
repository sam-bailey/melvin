# Introduction

This is the documentation for the melvin python package. This is a package which lets you do simple bayesian inference using maximum likelihood methods, built on top of Jax. These methods make some very strong assumptions, and are generally only good when you expect well-behaved posterior distributions, but these assumtpions allow them to run very quickly.