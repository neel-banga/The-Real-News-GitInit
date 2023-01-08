import viewpoints
import site_tester
import sys

print('---- Welcome ------')


choice = int(input("Would you like to explore a certain topic (1) or check your website's baises?(2) \n Type any other number to kill the program. \n"))

while True:
    if choice == 1:
        viewpoints.get_topic()

    elif choice == 2:
        site_tester.get_link_site()
    
    else:
        sys.exit()