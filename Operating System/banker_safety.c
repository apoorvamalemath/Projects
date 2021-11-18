#include<stdio.h>
#include<stdlib.h>
int main()
{
int allocation[10][10],need[10][10],avail[10],i,j,x,n,m,mmm,safe[10],process[10];
printf("\nEnter number of processes ");
scanf("%d",&n);
x=0;
int flag=0;
mmm=0;
for(i=0;i<n;i++)
process[i]=i+1;
printf("\nEnter the number of resources ");
scanf("%d",&m);
printf("\nEnter the allocation matrix");
for(i=0;i<m;i++)
for(j=0;j<n;j++)
scanf("%d",&allocation[i][j]);
printf("\nEnter the need matrix");
for(i=0;i<m;i++)
for(j=0;j<n;j++)
scanf("%d",&need[i][j]);
printf("\nEnter the available ");
for(i=0;i<m;i++)
scanf("%d",&avail[i]);
int work[m],finish[n];
for(i=0;i<m;i++)
work[i]=avail[i];
for(i=0;i<n;i++)
finish[i]=0;
l1:
for(i=0;i<n;i++)
{
if(finish[i]==0)
{
for(j=0;j<m;j++)
if(need[i][j]<=work[j])
{
mmm=1;
}
else
mmm=0;
if(mmm==1)
{
for(j=0;j<m;j++)
work[j]=work[j]+allocation[i][j];
safe[x]=process[i];
x++;
process[i]=0;
finish[i]=1;
}
}
}
for(i=0;i<n;i++)
if(process[i]!=0)
goto l1;
for(i=0;i<n;i++)
{
if(finish[i]==0)
{
printf("\nThe system is not in safe state");
flag=1;
break;
}
}
if(flag==0)
{
printf("\nThe system is in safe state");
for(i=0;i<n;i++)
printf(" %d",safe[i]);
}
return 0;
}
