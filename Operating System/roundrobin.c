
#include<stdio.h>
#include<stdlib.h>
int main()
{
int i,j,sum1[5]={0,0,0,0,0};
float awt=0,atat=0;
int process[5]={1,2,3,4,5};
int key=2;
int sum=0;
int x[5]={0,0,0,0,0};
int burst[5]={2,4,4,4,2};
int llll[5];
int wt[5]={0,0,0,0,0};
int tat[5]={0,0,0,0,0};
for(i=0;i<5;i++)
llll[i]=burst[i];
l1:
i=0;
while(i!=5)
{
if(burst[i]!=0)
{
if(burst[i]<=key)
{
sum=sum+burst[i];
sum1[i]=sum-burst[i];
burst[i]=0;

printf("p%d  %d\n",process[i],burst[i]);
}
else
{
sum=sum+key;
sum1[i]=sum+key;
x[i]=x[i]+key;
burst[i]=burst[i]-key;

printf("p%d  %d\n",process[i],burst[i]);
}
}
i++;
}
for(i=0;i<5;i++)
{
if(burst[i]!=0)
goto l1;
}
for(i=0;i<5;i++)
wt[i]=sum1[i]-x[i];
for(i=0;i<5;i++)
tat[i]=wt[i]+llll[i];
printf("\nprocess bursttime waitingtime turnaroundtime");
for(i=0;i<5;i++)
printf("\n p%d       %d       %d          %d",process[i],llll[i],wt[i],tat[i]);
printf("\n");
for(i=0;i<5;i++)
{
  awt=awt+wt[i];
   atat=atat+tat[i];
}
awt=awt/5;
atat=atat/5;
printf("avg waiting time is %f",awt);
printf("\navg turn around time is %f",atat);
return 0;
}
