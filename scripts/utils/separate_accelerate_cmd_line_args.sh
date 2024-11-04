parse_args() {
    # Initialize arrays for accelerate and training arguments
    accel_args=()
    train_args=()

    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --accel_*)
                # Strip the --accel_ prefix before adding to accel_args
                stripped_arg="${1/--accel_/--}"  # Remove the prefix
                accel_args+=("$stripped_arg" "$2") 
                shift  # Skip to next argument after value
                ;;
            *)
                train_args+=("$1")  # All other arguments go to train_args
                ;;
        esac
        shift  # Move to the next argument
    done

    # Return values by using 'declare -p' to print arrays to be captured
    declare -p accel_args train_args
}