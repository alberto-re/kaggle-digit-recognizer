#!/bin/sh

# The name of the bucket where CloudFormation template will be stored
S3_BUCKET="<BUCKET_NAME>"

# The name of the IAM role with S3 read access to the above bucket
IAM_ROLE="<ROLE_NAME>"

# The key pair to use for authentication
KEY_NAME="<KEY_NAME>"

# The name to be attributed to the stack
CLOUDFORMATION_STACK_NAME="deeplearning"

CLOUDFORMATION_TPL="deeplearning_ec2.json"

upload_tpl_if_changed() {
	aws s3 sync \
		--exclude "*" \
		--include $CLOUDFORMATION_TPL \
		. s3://$S3_BUCKET
}

validate_tpl() {
	aws cloudformation validate-template \
		--template-url https://$S3_BUCKET.s3.amazonaws.com/$CLOUDFORMATION_TPL
	if [[ "$?" -eq "0" ]]; then
		echo Template not valid, launch aborted
		exit 255
	fi
}

# Returns 0 if true, otherwise 1
has_stack_been_created() {
	aws cloudformation list-stacks \
		--stack-status-filter CREATE_COMPLETE CREATE_IN_PROGRESS \
		--query StackSummaries[*].StackName \
		--output text \
		| grep $CLOUDFORMATION_STACK_NAME > /dev/null
}

create_stack() {
	upload_tpl_if_changed
	#validate_tpl
	aws cloudformation create-stack \
		--stack-name $CLOUDFORMATION_STACK_NAME \
		--template-url https://$S3_BUCKET.s3.amazonaws.com/$CLOUDFORMATION_TPL \
		--parameters ParameterKey=IamInstanceProfileParameter,ParameterValue=$IAM_ROLE ParameterKey=KeyNameParameter,ParameterValue=$KEY_NAME \
		> /dev/null
}

delete_stack() {
	aws cloudformation delete-stack \
		--stack-name $CLOUDFORMATION_STACK_NAME
}

get_ec2_public_ip() {
	aws ec2 describe-instances \
		--filters "Name=tag-key,Values=aws:cloudformation:stack-name" \
			"Name=tag-value,Values=$CLOUDFORMATION_STACK_NAME" \
		--filters "Name=instance-state-name,Values=running" \
		--query Reservations[0].Instances[0].PublicIpAddress \
		--output text
}

case $1 in
launch)
	if ! has_stack_been_created; then
		create_stack
		echo Instance launched
	else
		echo Instance already launched
	fi
	;;
halt)
	if has_stack_been_created; then
		delete_stack
	else
		echo Instance not launched
	fi;
	;;
status)
	if has_stack_been_created; then
		echo Instance launched or in progress \(IP $(get_ec2_public_ip)\)
	else
		echo Instance not launched
	fi
	;;
*)
	echo Valid actions are: launch, halt, status
	;;
esac
