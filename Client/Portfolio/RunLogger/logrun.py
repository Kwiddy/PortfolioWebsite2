from datetime import datetime, date

def ns_input():
    ns = input("Would you like to enter a new run [n] or view stats [s]? ")
    if ns.upper() != "N" and ns.upper() != "S":
        print("ERROR - Input must be an 'n' or 's'")
        ns_input()
    else:
        return ns.upper()

def main():
    print("-------------------------")
    print("WELCOME TO THE RUN LOGGER")
    print("-------------------------")

    ns = ns_input()

    if ns == "N":
        input_newrun()
    else:
        show_stats()
    
    print()
    yn = input("Would you like to input another command [y/n]? ")
    if yn.upper() == "Y":
        print()
        main()
    else:
        exit()

def input_newrun():
    print()
    print("-----------------")
    print("LOGGING A NEW RUN")
    print("-----------------")
    print()

    distance = input("Distance (km): ")
    time = input("Time taken (min:sec): ")
    hipStat = input("How did your hip/knee feel: ")
    rt = input("Road [r], Treadmill [t], or both [b]? ")

    if rt.upper() == "R":
        rt = "Road"
    elif rt.upper() == "T":
        rt = "Treadmill"
    elif rt.upper() == "B":
        rt = "Both"

    f = open("runlog.txt", "a+")

    now = datetime.now()
    d_string = now.strftime("%d-%m-%Y")

    inString = d_string + "\t" + distance + "\t" + time + "\t" + rt + "\t" + hipStat + "\n"

    f.write(inString)
    f.close()

def show_stats():
    try:
        f = open("runlog.txt", "r")
        print()

        store = []
        for line in f:
            store.append(line.split("\t"))
        f.close()

        distance = 0
        for item in store:
            distance += float(item[1])

        now = datetime.now()

        day = now.strftime("%d")
        month = now.strftime("%m")
        year = now.strftime("%Y")

        f_date = date(int(year), int(month), int(day))
        l_date = date(2020, 12, 31)
        delta = l_date - f_date

        remaining1 = 180-(distance/1.6)
        remaining2 = 365-(distance/1.6)

        weekNum = f_date.isocalendar()[1]

        print("Number of runs: " + str(len(store)))
        print("Total distance: " + str(round(distance/1.6)) + " miles (" + str(round(distance)) + "km)")
        print("Distance remaining (180 miles): " + str(round(180-(distance/1.6))) + " miles (" + str((remaining1/delta.days)) + " miles per day)")
        print("Distance remaining (365 miles): " + str(round(365-(distance/1.6))) + " miles (" + str((remaining2/delta.days)) + " miles per day)")
        print("Week Number: " + str(weekNum))
        print("Avg. number of runs per week: " + str((len(store)/weekNum)))
        print("Avg. number of miles per week: " + str(((distance/1.6)/weekNum)))

        print()
        yn = input("Would you like to view the runs so far [y/n]? ")
        if yn.upper() == "Y":
            print()
            print()
            print("-------------------------------------------------------------")
            print("Date     Distance(km)   Time    Route           Details")
            print("-------------------------------------------------------------")
            print()
            f = open("runlog.txt", "r")
            for line in f:
                if "Road" in line:
                    myString = line.split("\t")
                    i = 0
                    while i< len(myString):
                        if myString[i] == "Road":
                            myString.insert(i+1, " ")
                        i += 1
                    line = "\t".join(myString)
                    print(line)
                elif "Both" in line:
                    print()
                    f = open("runlog.txt", "r")
                    for line in f:
                        if "Road" in line:
                            myString = line.split("\t")
                            i = 0
                            while i< len(myString):
                                if myString[i] == "Road":
                                    myString.insert(i+1, " ")
                                i += 1
                            line = "\t".join(myString)
                            print(line)
                else:
                    print(line)
            f.close()

    except Exception as e:
        print(e)
        print("ERROR - No Log file exists")

main()