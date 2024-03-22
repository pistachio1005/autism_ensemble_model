#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import boto3
import os
import configparser
import json

# Read AWS credentials from configuration file
config = configparser.ConfigParser()
config.read('config_instance.ini')

# Function to get EC2 resource object for interacting with AWS EC2
def get_ec2_resource(region_name="us-east-1"):
    ec2_resource = boto3.resource("ec2",
                                    region_name=region_name,
                                    aws_access_key_id=config['AWS']['ACCESS_KEY'],
                                    aws_secret_access_key=config['AWS']['SECRET_KEY'])
    return ec2_resource

# Function to get EC2 client object for making API calls to AWS EC2
def get_ec2_client(region_name="us-east-1"):
    ec2_client = boto3.client("ec2",
                                region_name=region_name,
                                aws_access_key_id=config['AWS']['ACCESS_KEY'],
                                aws_secret_access_key=config['AWS']['SECRET_KEY'])
    return ec2_client

# Function to create a key pair for EC2 instances if it doesn't already exist
def create_key_pair_if_not_exists(key_name="ec2-key-pair", region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    key_pairs = ec2_client.describe_key_pairs()["KeyPairs"]
    key_pair_names = [key_pair["KeyName"] for key_pair in key_pairs]
    if key_name not in key_pair_names:
        path = create_key_pair(key_name, region_name)
        return path
    else:
        print("Key pair already exists. Trying to load it from file.")
        path = "aws_" + key_name + ".pem"
        if os.path.exists(path):
            print("Key pair file exists. Loading it.")
            return path
        else:
            raise Exception("Key pair file does not exist. Please create it manually.")

# Function to actually create a key pair in AWS and save the private key locally
def create_key_pair(key_name="ec2-key-pair", region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    key_pair = ec2_client.create_key_pair(KeyName=key_name)

    private_key = key_pair["KeyMaterial"]

    # Write private key to a file with secure permissions
    path = "aws_" + key_name + ".pem"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT, 0o700), "w+") as handle:
        handle.write(private_key)
    return path

# Function to retrieve the public IP address of a specified EC2 instance
def get_public_ip(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    reservations = ec2_client.describe_instances(InstanceIds=[instance_id]).get("Reservations")

    ip_addresses = []
    for reservation in reservations:
        for instance in reservation['Instances']:
            ip_addresses.append(instance.get("PublicIpAddress"))
    return ip_addresses

# Function to start a specified EC2 instance
def start_instance(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    response = ec2_client.start_instances(InstanceIds=[instance_id])
    print(response)

# Function to stop a specified EC2 instance
def stop_instance(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    response = ec2_client.stop_instances(InstanceIds=[instance_id])
    print(response)

# Function to terminate a specified EC2 instance
def terminate_instance(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    response = ec2_client.terminate_instances(InstanceIds=[instance_id])
    print(response)

# Function to save instance IP addresses to a JSON file
def save_instance_ips(instance_ips):
    with open("ips.json", 'w') as f:
        f.write(json.dumps(instance_ips))

# Function to load instance IP addresses from a JSON file
def load_instance_ips():
    with open("ips.json", 'r') as f:
        instance_ips = json.loads(f.read())
    return instance_ips
