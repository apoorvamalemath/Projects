#include<stdio.h>
#include<stdlib.h>
int main()
{
int i,j,sum=0;
float awt=0,atat=0;
int wt[5],tat[5];
int priority[5]={0,4,3,1,2};
int bursttime[5]={20,5,10,7,3};
for(i=0;i<5-1;i++)
{
for(j=0;j<5-i-1;j++)
if(priority[j]>priority[j+1])
{
int t,x;
t=priority[j];
priority[j]=priority[j+1];
priority[j+1]=t;
x=bursttime[j];
bursttime[j]=bursttime[j+1];
bursttime[j+1]=x;
}

}
wt[0]=0;
for(i=1;i<5;i++)
{
sum=0;
for(j=0;j<i;j++)
sum=sum+bursttime[j];
wt[i]=sum;
}
for(i=0;i<5;i++)
{
sum=0;
for(j=0;j<=i;j++)
sum=sum+bursttime[j];
tat[i]=sum;
}
for(i=0;i<5;i++)
awt=awt+wt[i];
for(i=0;i<5;i++)
atat=atat+tat[i];
awt=awt/5;
atat=atat/5;
printf("\nbursttime   waitingt  turnaroundt\n");
for(i=0;i<5;i++)
printf("\n%d      %d         %d         %d",priority[i],bursttime[i],wt[i],tat[i]);
printf("\nAverage waiting time is %.2f",awt);
printf("\nAverage turn around time is %.2f",atat);
return 0;
}
